
# ScaledYOLOv4 with PyTorch Mish

The original [ScaledYOLOv4-large branch](https://github.com/WongKinYiu/ScaledYOLOv4 "ScaledYOLOv4-large branch") repo uses [mish-cuda library](https://github.com/thomasbrandon/mish-cuda "mish-cuda library") for [mish activation function](https://arxiv.org/abs/1908.08681 "mish activation function"). 
When using detect.py and test.py with cuda-mish trained weights, non-CUDA device (CPU etc..) encounter some errors like *missing cuda-mish*.

This library uses native [PyTorch mish class](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html "Pytorch mish class") instead of mish-cuda, like [WongKinYiu's yolor layer.py](https://github.com/WongKinYiu/yolor/blob/main/utils/layers.py "WongKinYiu's yolor layer.py"). So it can be used in cases where CPU is used for detecting or CPU training.
WARNING: torch.nn.Mish class only available in PyTorch 1.9.0 and upper versions.

# Performance Tests for Training

Used Library | Training Time | Recorded Notebook
------------- | ------------- | -------------
ScaledYOLOv4 with mish-cuda @ PyTorch 1.9.0 | 90 epochs completed in 6.702 hours | <a href="https://www.kaggle.com/mesih5/mikro-basic-aug-scaledyolov4-largebranch-torcmish?scriptVersionId=67554906"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
ScaledYOLOv4 with PyTorch.nn.mish @ PyTorch 1.9.0 | 95 epochs completed in 6.931 hours | <a href="https://www.kaggle.com/mesih5/mikro-basic-aug-scaledyolov4-largebranch-mishcuda?scriptVersionId=67617790"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>


# Acknowledgements

- https://github.com/WongKinYiu/PyTorch_YOLOv4
- https://pytorch.org/docs/stable/generated/torch.nn.Mish.html