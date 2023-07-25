# centerpoint-livox
CenterPoint model trained on livox dataset, and deployed with TensorRT on ros2
本仓库是将https://github.com/Livox-SDK/livox_detection.git   仓库中python torch工程进行了tensorrt部署，目前在3060笔记本电脑上耗时为25ms左右。

工作思路：
  pth转onnx转tensorrt；
存在问题：
  1、模型转onnx，atan2算子onnx不支持；
  2、mask操作带来Gather和NoZero操作；
  3、mod算子tensorrt不支持；
  4、atan2算子tensorrt不支持；
解决方法：
  1、模型生成时添加 operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK，先生成模型，后续再解决tensorrt不支持的问题；
  2、将mask写在后处理cuda c中；
  3、编写mod trt算子；
  4、编写atan2 trt算子。
