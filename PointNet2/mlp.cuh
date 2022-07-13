#include <cudnn.h>
#include <cuda_runtime.h>
#include <cassert>
#include "algorithm"

using algo_perf_t = cudnnConvolutionFwdAlgoPerf_t;

#ifdef CUDA_ERROR_CHECK
#define checkCUDNN(expression)                                  \
  {                                                             \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
	    std::cerr << "Error on line " << __LINE__ << ": "       \
	    << cudnnGetErrorString(status) << std::endl;            \
	    std::exit(EXIT_FAILURE);                                \
    }                                                           \
 }
#else
#define checkCUDNN(expression)                                  \
  {                                                             \
    (expression);                                               \
  }
#endif

bool get_valid_best_algo(std::vector<algo_perf_t>& algo_arr) {
    auto it = std::remove_if(algo_arr.begin(), algo_arr.end(), [](algo_perf_t algo_perf){
        return algo_perf.status != CUDNN_STATUS_SUCCESS;
    });
    algo_arr.erase(it, algo_arr.end());
    if(algo_arr.size() == 0) {
        std::runtime_error("Found no valid conv algorithm!");
    }
    std::sort(algo_arr.begin(), algo_arr.end(), [](algo_perf_t algo1, algo_perf_t algo2){
        return algo1.time < algo2.time;
    });
    return algo_arr.size()>0;
}


void mlp(float* d_input, float* d_output, int cov_high, int cov_width, int input_channel, int output_channel, int input_high, int input_width){

    cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
 
	// 输入张量的描述
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,	// 注意是 NHWC，TensorFlow更喜欢以 NHWC 格式存储张量(通道是变化最频繁的地方，即 BGR)，而其他一些更喜欢将通道放在前面
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/input_channel,
		/*image_height=*/input_high,
		/*image_width=*/input_width));
 
	// 卷积核的描述（形状、格式）
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,	// 注意是 NCHW
		/*out_channels=*/output_channel,
		/*in_channels=*/input_channel,
		/*kernel_height=*/cov_high,
		/*kernel_width=*/cov_width));
#ifdef CUDA_DEBUG_
    std::cerr << "kernel: " << input_channel << "->" << output_channel << "[" << cov_high <<" x " << cov_width << " ] " << std::endl;
#endif
    // 卷积操作的描述（步长、填充等等）
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/0,
		/*pad_width=*/0,
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CONVOLUTION, // CUDNN_CONVOLUTION
		/*computeType=*/CUDNN_DATA_FLOAT));
 
	// 计算卷积后图像的维数
	int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
		input_descriptor,
		kernel_descriptor,
		&batch_size,
		&channels,
		&height,
		&width));

#ifdef CUDA_DEBUG_
	std::cerr << "Output Image: " << height << " x " << width << " x " << channels
		<< std::endl;
#endif

	// 卷积输出张量的描述
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/output_channel,
		/*image_height=*/input_high,
		/*image_width=*/input_width));
 
	// 卷积算法的描述
	// cudnn_tion_fwd_algo_gemm——将卷积建模为显式矩阵乘法，
	// cudnn_tion_fwd_algo_fft——它使用快速傅立叶变换(FFT)进行卷积或
	// cudnn_tion_fwd_algo_winograd——它使用Winograd算法执行卷积。
    int request_cnt = 0;

    int algo_count = 0;
    std::vector<algo_perf_t> algo_perf_arr;

    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &request_cnt));
    algo_perf_arr.resize(request_cnt);

    checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
        request_cnt,
		/*memoryLimitInBytes=*/&algo_count,
        algo_perf_arr.data()));

    if(!get_valid_best_algo(algo_perf_arr)) {
        std::runtime_error("Found no valid conv algorithm!");
    }
    cudnnConvolutionFwdAlgo_t convolution_algorithm = algo_perf_arr[0].algo;

	// 计算 cuDNN 它的操作需要多少内存
	size_t workspace_bytes{ 0 };
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes));
#ifdef CUDA_DEBUG_
	std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
		<< std::endl;
	assert(workspace_bytes > 0);
#endif
 
	// *************************************************************************
	// 分配内存， 从 cudnnGetConvolutionForwardWorkspaceSize 计算而得
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);
 


	// *************************************************************************
	// clang-format off
	const float kernel_template[1][1] = {
		{ 0.6 }
	};
	// clang-format on
    
	float h_kernel[output_channel][input_channel][1][1]; // NCHW
	for (int kernel = 0; kernel < output_channel; ++kernel) {
		for (int channel = 0; channel < input_channel; ++channel) {
			for (int row = 0; row < 1; ++row) {
				for (int column = 0; column < 1; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}
 
	float* d_kernel{ nullptr };
	cudaMalloc(&d_kernel, sizeof(h_kernel));
	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
	// *************************************************************************
 
	const float alpha = 1.0f, beta = 0.0f;
 
	// 真正的卷积操作 ！！！前向卷积
	checkCUDNN(cudnnConvolutionForward(cudnn,
		&alpha,
		input_descriptor,
		d_input,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm,
		d_workspace, // 注意，如果我们选择不需要额外内存的卷积算法，d_workspace可以为nullptr。
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output));
 
	cudaFree(d_kernel);
	cudaFree(d_workspace);
 
	// 销毁
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
 
	cudnnDestroy(cudnn);

}
