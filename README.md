# Bucket-based Farthest point sampling for largest-scaled point clouds

![](https://komarev.com/ghpvc/?username=hanm2019bfpsGPU)

we use an approximate KD-Tree to divide the point clouds into multi-buckets and use two geometry inequality to reduce the distance computation times and the data which need to load from memory

we present the GPU implementation and [CPU implementation](https://github.com/hanm2019/FPS_CPU) of bucket-based farthest point sampling.

The GPU implementation is tested in NVIDIA 1080Ti, 2080Ti, AGX Xavier.

Note that, when executing on AGX Xavier, you need to adjustments the `numOfCudaCores` value ( to 256 maybe) in `src/kdtree.cuh`

# BUILD

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

```

 then, two executable files are generated:

* FPS_Baseline_GPU: the conventional implementation of  FPS, used for performance baseline.
* FPS_KDtree_GPU: bucket-based farthest point sampling, each bucket contains multiple points.  **high performance** 

  

# USAGE

```
./FPS_Baseline_GPU  num_sample_point filename
./FPS_KDtree_GPU tree_high num_sample_point filename
```

# Cite

Please kindly consider citing this repo in your publications if it helps your research.

```
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

# Resources

1. [the CPU implementation of FPS](https://github.com/hanm2019/FPS_CPU)
2. the reference paper: [QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds](https://ieeexplore.ieee.org/abstract/document/10122654)
