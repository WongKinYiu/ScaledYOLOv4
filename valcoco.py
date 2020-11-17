import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == '__main__':
    pred = sys.argv[1]
    gt = 'instances_val2017.json'

    cocoGt = COCO(gt)
    cocoDt = cocoGt.loadRes(pred)

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
