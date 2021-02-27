from pycocotools.cocoeval import COCOeval
import json
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from colorama import Fore, Style
# from bounding_box import bounding_box as bb

from retinanet.dataloader import eval_collate
from retinanet.utils import get_logger

logger = get_logger(__name__, level="info")


def evaluate_coco(
    dataset,
    val_image_ids,
    logdir
    # dataset, model, logdir, batch_size, num_workers, writer=None, n_iter=None, threshold=0.05
):

    

    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes(os.path.join(logdir, "val_bbox_results.json"))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, "bbox")
    coco_eval.params.imgIds = val_image_ids
    coco_eval.params.iouThrs = np.array([0.4])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    # model.train()

    return stats
