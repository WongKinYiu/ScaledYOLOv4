#ScaledYOLOv4 with PyTorch Mish
The original [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4 "ScaledYOLOv4") repo uses [mish-cuda library](https://github.com/thomasbrandon/mish-cuda "mish-cuda library") for [mish activation function](https://arxiv.org/abs/1908.08681 "mish activation function"). 
When using detect.py and test.py with cuda-mish trained weights, non-CUDA device (CPU etc..) encounter some errors like *missing cuda-mish*.

This library uses native [PyTorch mish class](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html "Pytorch mish class") instead of mish-cuda. So it can be used in cases where CPU is used for detecting.