# YOLOv4-large

This is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" using PyTorch framwork.

* [YOLOv4-CSP](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp)
* [YOLOv4-tiny](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-tiny)
* [YOLOv4-large](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large)

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4-P5** | 896 | **51.4%** | **69.9%** | **56.3%** | **33.1%** | **55.4%** | **62.4%** | 41 *fps* |
| **YOLOv4-P5** | TTA | **52.5%** | **70.3%** | **58.0%** | **36.0%** | **52.4%** | **62.3%** | - |
|  |  |  |  |  |  |  |
| **YOLOv4-P6** | 1280 | **54.3%** | **72.3%** | **59.5%** | **36.6%** | **58.2%** | **65.5%** | 30 *fps* |
| **YOLOv4-P6** | TTA | **54.9%** | **72.6%** | **60.2%** | **37.4%** | **58.8%** | **66.7%** | - |
|  |  |  |  |  |  |  |
| **YOLOv4-P7** | 1536 | **55.4%** | **73.3%** | **60.7%** | **38.1%** | **59.5%** | **67.4%** | 15 *fps* |
| **YOLOv4-P7** | TTA | **55.8%** | **73.2%** | **61.2%** | **38.8%** | **60.1%** | **68.2%** | - |
|  |  |  |  |  |  |  |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv4-P5** | 896 | **51.2%** | **69.8%** | **56.2%** | **35.0%** | **56.2%** | **64.0%** | [`yolov4-p5.pt`](https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing) |
| **YOLOv4-P5** | TTA | **52.5%** | **70.2%** | **57.8%** | **38.5%** | **57.2%** | **64.0%** | - |
| **YOLOv4-P5** (+BoF) | 896 | **51.7%** | **70.3%** | **56.7%** | **35.9%** | **56.7%** | **64.3%** | [`yolov4-p5_.pt`](https://drive.google.com/file/d/15CL05ZufFk2krbRS993fqlG40Wb0HTyr/view?usp=sharing) |
| **YOLOv4-P5** (+BoF) | TTA | **52.8%** | **70.6%** | **58.3%** | **38.8%** | **57.4%** | **64.4%** | - |
|  |  |  |  |  |  |  |  |
| **YOLOv4-P6** | 1280 | **53.9%** | **72.0%** | **59.0%** | **39.3%** | **58.3%** | **66.6%** | [`yolov4-p6.pt`](https://drive.google.com/file/d/1aB7May8oPYzBqbgwYSZHuATPXyxh9xnf/view?usp=sharing) |
| **YOLOv4-P6** | TTA | **54.4%** | **72.3%** | **59.6%** | **39.8%** | **58.9%** | **67.6%** | - |
| **YOLOv4-P6** (+BoF) | 1280 | **54.4%** | **72.7%** | **59.5%** | **39.5%** | **58.9%** | **67.3%** | [`yolov4-p6_.pt`](https://drive.google.com/file/d/1Q8oG3lBVVoS0-UwNOBsDsPkq9VKs9UcC/view?usp=sharing) |
| **YOLOv4-P6** (+BoF) | TTA | **54.8%** | **72.6%** | **60.0%** | **40.6%** | **59.1%** | **68.2%** | - |
| **YOLOv4-P6** (+BoF*) | 1280 | **54.7%** | **72.9%** | **60.0%** | **39.4%** | **59.2%** | **68.3%** |  |
| **YOLOv4-P6** (+BoF*) | TTA | **55.3%** | **73.2%** | **60.8%** | **40.5%** | **59.9%** | **69.4%** | - |
|  |  |  |  |  |  |  |  |
| **YOLOv4-P7** | 1536 | **55.0%** | **72.9%** | **60.2%** | **39.8%** | **59.9%** | **68.4%** | [`yolov4-p7.pt`](https://drive.google.com/file/d/18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3/view?usp=sharing)  |
| **YOLOv4-P7** | TTA | **55.5%** | **72.9%** | **60.8%** | **41.1%** | **60.3%** | **68.9%** | - |
|  |  |  |  |  |  |  |  |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv4-P6-attention** | 1280 | **54.3%** | **72.3%** | **59.6%** | **38.7%** | **58.9%** | **66.6%** |

## Installation

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3

# install mish-cuda, if you use different pytorch version, you could try https://github.com/thomasbrandon/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

## Testing

```
# download {yolov4-p5.pt, yolov4-p6.pt, yolov4-p7.pt} and put them in /yolo/weights/ folder.
python test.py --img 896 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p5.pt
python test.py --img 1280 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p6.pt
python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p7.pt
```

You will get following results:
```
# yolov4-p5
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51244
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69771
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.56180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.56247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.63983
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.64048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.69801
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.55487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.74368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.82826
```
```
# yolov4-p6
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.53857
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.72015
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.59025
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.39285
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.58283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.39552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.66504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.72141
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.59193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.75844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83981
```
```
# yolov4-p7
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.55046
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.72925
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.60224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.39836
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.59854
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.68405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.40256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.66929
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.72943
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.59943
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.76873
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84460
```

## Training

We use multiple GPUs for training.
{YOLOv4-P5, YOLOv4-P6, YOLOv4-P7} use input resolution {896, 1280, 1536} for training respectively.
```
# yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights '' --sync-bn --device 0,1,2,3 --name yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights 'runs/exp0_yolov4-p5/weights/last_298.pt' --sync-bn --device 0,1,2,3 --name yolov4-p5-tune --hyp 'data/hyp.finetune.yaml' --epochs 450 --resume
```

If your training process stucks, it due to bugs of the python.
Just `Ctrl+C` to stop training and resume training by:
```
# yolov4-p5
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights 'runs/exp0_yolov4-p5/weights/last.pt' --sync-bn --device 0,1,2,3 --name yolov4-p5 --resume
```

## Citation

```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>
