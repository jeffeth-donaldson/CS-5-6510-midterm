
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Give metrics on relative segmentation/classification quality
# comparing Mask R-CNN, faster R-CNN, and RetinaNet.

# Model zoo locations
# Faster R-CCN - COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
# RetinaNet - COCO-Detection/retinanet_R_50_FPN_1x.yaml
# Mask R-CNN - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

# TO run with a different model replace on lines 26 and 29

# Mask R-CNN
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
test_path = 'data/grocerystore'
validation_img_paths = [f'{test_path}/{f}' for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))][:100]
validation_imgs = [cv2.imread(f) for f in validation_img_paths]
for im in validation_imgs:
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(out.get_image()[:, :, ::-1])
