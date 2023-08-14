# centerpoint-livox: An Lidar Object Detection project implemented by TensorRT
CenterPoint model trained on livox dataset and deployed with TensorRT on ros2. Code is written according to this [project](https://github.com/Livox-SDK/livox_detection.git).

Overall inference has three phases:
* Conver points cloud into boolmap;  
  boolmap: use true or fasle to describe voxels  
  (N,5)-->(1,30,448,1120);  5D: index in one batch, x, y, z, i  
* Run rpn backbone TensorRT engine to get 3D-detection raw data  
  (1,30,448,1120)-->box:(500,7),score:(500),label:(500)  
* do mask and nms on cuda c to get filtered output  
  box:(500,7),score:(500),label:(500)-->output_box:(num,7),output_score:(num),output_label:(num)  
  
# Data
the project is running inference on [LivoxOpenDataSet](https://www.livoxtech.com/cn/dataset).

# Model
The .pt file you can get in this [project](https://github.com/Livox-SDK/livox_detection.git). the onnx file is provided by this project.
You can also get the onnx file by yourself through programming, but you have to consider the problem that onnx does not support atan2.

# Prerequisites
TensorRT and cuda are necessary conditions for running centerpoint

# Environments
* Nvidia RTX 3060 Laptop GPU  
* Cuda11 + cuDNN8 + tensorrt8
* ws_msgs here: https://github.com/Tream733/ws_msgs

# Performance in FP16
```
| Function(unit:ms) | NVIDIA RTX 3060 Laptop GPU | NVIDIA Jetson AGX Orin      |
| ----------------- | --------------------------- | --------------------------- |
| Preprocess        | 0.044534 ms                 | 0.323782  ms                 |
| Rpn               | 25.0291  ms                 | 48.5475  ms                 |
| Postprocess       | 0.80967 ms                  | 1.8511  ms                 |
| Summary           | 25.8914  ms                 | 50.7353  ms                 |
```
# Visualization
![Visualization](output.gif)

# References
- [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)
- [mmdetection3d](https://github.com/Tartisan/mmdetection3d)
- [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint)
- [livox-SDK/livox_detection](https://github.com/Livox-SDK/livox_detection.git)

