#!/bin/bash
# Used for training with 3 GTX 1070 GPUs
python -m torch.distributed.launch --nproc_per_node 3 train.py --device 0,1,2 --batch-size 21 --data speedco.yaml --weights '' --cfg yolov4-csp.cfg --name yolov4-csp-speedco --sync-bn --rect --single-cl