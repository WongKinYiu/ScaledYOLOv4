# YOLOv4-tiny

This is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" using Darknet framwork.

The implementation is supported by [Darknet](https://github.com/AlexeyAB/darknet), just use it.

## Installation

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.02-py3

# install opencv
apt update
apt install libopencv-dev

# go to code folder
cd /yolo
make -j4
```

## Testing

[`yolov4-tiny.weights`](https://drive.google.com/file/d/1XLVy_DMjvhhmHucSypL3zeDGDdmCGrsu/view?usp=sharing)

```
# download yolov4-tiny.weights and put it in /yolo/weights/ folder.
./darknet detector valid cfg/coco.data cfg/yolov4-tiny.cfg weights/yolov4-tiny.weights -out yolov4-tiny -gpus 0
python valcoco.py ./results/yolov4-tiny.json
```

You will get the results:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.102
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
```

## Training

```
./darknet detector train cfg/coco.data cfg/yolov4-tiny.cfg -gpus 0 -dont_show
```

For resume training:
```
# assume the checkpoint is stored in ./coco/.
./darknet detector train cfg/coco.data cfg/yolov4-tiny.cfg coco/yolov4-tiny_last.weights -gpus 0 -dont_show
```

## Citation

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
