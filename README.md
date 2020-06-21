# MMML
This is a matlab implementation of the following paper:

R. Wang, X. Wu, K. Chen and J. Kittler, "Multiple Manifolds Metric Learning with Application to Image Set Classification," 2018 24th International Conference on Pattern Recognition (ICPR), Beijing, 2018, pp. 627-632, doi: 10.1109/ICPR.2018.8546030.

In this folder:
  (1) the main.m is the core file of this code, you can run it to start this code.
  (2) the demo-ETH.mat is a running example, constructed from the ETH-80 datasets.
  (3) the compute_metric_learning.m is used to learn the target transformation matrix.
  (4) the compute_sub.m is used to generate the Grassmann manifold-valued data.
  (5) the compute_cov.m is used to generate the SPD manifold-valued data.
  
If you find this code is useful for your own work, please kindly cite the following:

    @INPROCEEDINGS{8546030,
      author={Rui Wang, Xiao-Jun Wu, Kai-Xuan Chen, Josef Kittler},
      booktitle={2018 24th International Conference on Pattern Recognition (ICPR)}, 
      title={Multiple Manifolds Metric Learning with Application to Image Set Classification}, 
      year={2018},
      pages={627-632},}

If you have any queries about this code, please do not hesitate to contact me at: cs_wr@jiangnan.edu.cn 
