"""
Useful helpers for coco datasets
"""

import os.path as osp
import json
from pathlib import Path
import warnings
from copy import deepcopy

import cv2
import numpy as np
from .bbox_utils import draw_bboxes

class CocoDataset():
    def __init__(self, json_file):
        self.root_dir = Path(json_file).parent
        with open(json_file, 'r') as f:
            self.data = json.loads(f.read())
        image_ids = self.get_image_ids()
        if len(image_ids) != len(set(image_ids)):
            warnings.warn("There is an issue! Image ids are not unique")
        ann_ids = self.get_ann_ids()
        if len(ann_ids) != len(set(ann_ids)):
            warnings.warn("There is an issue! Annotation ids are not unique")
        cat_ids = self.get_category_ids()
        if len(cat_ids) != len(set(cat_ids)):
            warnings.warn("There is an issue! Category ids are not unique")
        # Now rearrange data so that instead if lists we have id indexed dicts
        self.image_items = {}
        for image_row in deepcopy(self.data['images']):
            image_id = image_row.pop('id')
            self.image_items[image_id] = image_row
            self.image_items[image_id]['annotations'] = {}
            for ann_row in self.data['annotations']:
                if ann_row['image_id'] != image_id:
                    continue
                ann_id = ann_row.pop('id')
                self.image_items[image_id]['annotations'][ann_id] = ann_row
        self.categories = {
            cat_row['id']: cat_row['name'] for cat_row in self.data['categories']}


    def get_image_ids(self):
        return [row['id'] for row in self.data['images']]

    def get_ann_ids(self, image_ids=[]):
        """
        Use image_ids to filter
        """
        if len(image_ids):
            image_ids = set(image_ids)
            return [row['id'] for row in self.data['annotations']
                    if row['image_id'] in image_ids]
        else:
            return [row['id'] for row in self.data['annotations']]

    def get_category_ids(self):
        return [row['id'] for row in self.data['categories']]

    def render_anns(self, image_id, bboxes=True, masks=False):
        assert not masks, "masks rendering not implemented yet. TODO"
        image_item = self.image_items[image_id]
        img = cv2.imread(osp.join(self.root_dir, image_item['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = []
        category_ids = []
        for ann_row in image_item['annotations'].values():
            bboxes.append(ann_row['bbox'])
            category_ids.append(ann_row['category_id'])
        bboxes = np.array(bboxes)
        category_names = [
            self.categories[cat_id] for cat_id in category_ids]
        drawn = draw_bboxes(
            img, bboxes.astype(int), labels=category_names, labeled=True)
        return drawn