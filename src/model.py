import torch
import supervisely as sly
from supervisely.imaging.color import random_rgb
import src.sly_globals as g
from torchvision.ops import nms

import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.join(g.app_root_directory, "scenic"))
from scenic.projects.owl_vit.configs import clip_b16, clip_b32, clip_l14
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.notebooks import inference
from scenic.model_lib.base_models import box_utils

OBJECT_COLORS = {}

def apply_model(
        input_image: np.array, 
        model, 
        query_image: np.array = None, 
        reference_bbox = None, 
        text_queries = None, 
        class_name: str = None, 
        confidence_threshold: float = 0.5, 
        nms_threshold: float = 0.5
    ):
    # width, height, channels
    img_height, img_width = input_image.shape[:2]
    if query_image is not None:
        reference_image = query_image
        ref_img_height, ref_img_width = reference_image.shape[:2]
        bbox_coordinates = reference_bbox
        label = class_name
        # normalize bounding box coordinates to format required by tensorflow
        # image will be padded to squared form, so it is necessary to adapt bbox coordinates to padded image
        scaler = max(ref_img_height, ref_img_width)
        bbox_coordinates[0] = bbox_coordinates[0] / scaler
        bbox_coordinates[1] = bbox_coordinates[1] / scaler
        bbox_coordinates[2] = bbox_coordinates[2] / scaler
        bbox_coordinates[3] = bbox_coordinates[3] / scaler
        bbox_coordinates = np.array(bbox_coordinates)

        # pass reference image to model
        reference_embeddings, bbox_idx = model.embed_image_query(
            query_image=reference_image,
            # TensorFlow format (y_min, x_min, y_max, x_max), normalized to [0, 1]
            query_box_yxyx=bbox_coordinates, 
        )
        n_queries = 1  # model does not support multi-query image-conditioned detection
        # get model predictions
        top_query_idx, scores = model.get_scores(
            input_image,
            reference_embeddings[None, ...],
            num_queries=1,
        )
        _, _, input_image_boxes = model.embed_image(input_image)
        input_image_boxes = box_utils.box_cxcywh_to_yxyx(input_image_boxes, np)

        # apply nms to predicted boxes (scores of suppressed boxes will be set to 0)
        for i in np.argsort(-scores):
            if not scores[i]:
                # this box is already suppressed, continue:
                continue
            ious = box_utils.box_iou(
                input_image_boxes[None, [i], :],
                input_image_boxes[None, :, :],
                np_backbone=np)[0][0, 0]
            ious[i] = -1.0  # mask self-iou
            scores[ious > nms_threshold] = 0.0
        
        # postprocess model predictions
        new_annotation = sly.Annotation(img_size=(img_height, img_width))
        for box, score in zip(input_image_boxes, scores):
            if score >= confidence_threshold:
                # image was padded to squared form, so it is necessary to adapt bbox coordinates to padded image
                scaler = max(img_height, img_width) 
                box[0] = round(box[0] * scaler)
                box[1] = round(box[1] * scaler)
                box[2] = round(box[2] * scaler)
                box[3] = round(box[3] * scaler)
                score = round(float(score), 2)

                color = OBJECT_COLORS.get(label, random_rgb())
                if label not in OBJECT_COLORS.keys():
                    OBJECT_COLORS[label] = color
                obj_class = sly.ObjClass(label, sly.Rectangle, color=color)
                obj_label = sly.Label(sly.Rectangle(*box), obj_class)
                new_annotation = new_annotation.add_label(obj_label)
    else:
        # get text queries
        text_queries = tuple(text_queries)
        n_queries = len(text_queries)
        # extract embeddings from text queries
        query_embeddings = model.embed_text_queries(text_queries)
        # get box confidence scores
        top_query_ind, scores = model.get_scores(input_image, query_embeddings, n_queries)
        # extract input image features and get predicted boxes
        input_image_features, _, input_image_boxes = model.embed_image(input_image)
        input_image_boxes = box_utils.box_cxcywh_to_yxyx(input_image_boxes, np)
        
        # get predicted logits
        output = model._predict_classes_jitted(
            image_features=input_image_features[None, ...],
            query_embeddings=query_embeddings[None, ...],
        )
        # transform logits to labels
        labels = np.argmax(output['pred_logits'], axis=-1)
        labels = np.squeeze(labels) # remove unnecessary dimension
        # postprocess model predictions

        # apply nms to predicted boxes (scores of suppressed boxes will be set to 0)
        for i in np.argsort(-scores):
            if not scores[i]:
                # this box is already suppressed, continue:
                continue
            ious = box_utils.box_iou(
                input_image_boxes[None, [i], :],
                input_image_boxes[None, :, :],
                np_backbone=np)[0][0, 0]
            ious[i] = -1.0  # mask self-iou
            scores[ious > nms_threshold] = 0.0
            
        new_annotation = sly.Annotation(img_size=(img_height, img_width))
        for box, label, score in zip(input_image_boxes, labels, scores):
            if score >= confidence_threshold:
                # image was padded to squared form, so it is necessary to adapt bbox coordinates to padded image
                scaler = max(img_height, img_width) 
                box[0] = round(box[0] * scaler)
                box[1] = round(box[1] * scaler)
                box[2] = round(box[2] * scaler)
                box[3] = round(box[3] * scaler)
                label = text_queries[label]
                label = label.replace(" ", "_")
                label = label + "_pred"
                score = round(float(score), 2)

                color = OBJECT_COLORS.get(label, random_rgb())
                if label not in OBJECT_COLORS.keys():
                    OBJECT_COLORS[label] = color
                obj_class = sly.ObjClass(label, sly.Rectangle, color=color)
                obj_label = sly.Label(sly.Rectangle(*box), obj_class)
                new_annotation = new_annotation.add_label(obj_label)
    return new_annotation


def get_model(selected_model):
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # set GPU as visible device
        tf.config.set_visible_devices(gpus[0], "GPU")
    else:
        # hide GPUs from visible devices
        tf.config.set_visible_devices([], "GPU")
        
    # load selected model
    if selected_model == "OWL-ViT base patch 32":
        config = clip_b32.get_config(init_mode="canonical_checkpoint")
    elif selected_model == "OWL-ViT base patch 16":
        config = clip_b16.get_config(init_mode="canonical_checkpoint")
    elif selected_model == "OWL-ViT large patch 14":
        config = clip_l14.get_config(init_mode="canonical_checkpoint")
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )
    variables = module.load_variables(config.init_from.checkpoint_path)
    model = inference.Model(config, module, variables)
    model.warm_up()
    return model

def inference_json_anno_preprocessing(ann, project_meta: sly.ProjectMeta) -> sly.Annotation:
    temp_meta = project_meta.clone()
    pred_classes = []
    for i, obj in enumerate(ann['annotation']['objects']):
        class_ = obj['classTitle'] + '_pred'
        pred_classes.append(class_)
        ann['annotation']['objects'][i]['classTitle'] = class_
        
        if temp_meta.get_obj_class(class_) is None:
            new_obj_class = sly.ObjClass(class_, sly.Rectangle)
            temp_meta = temp_meta.add_obj_class(new_obj_class)
    if temp_meta.get_tag_meta('confidence') is None:
        temp_meta = temp_meta.add_tag_meta(sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER))
    return sly.Annotation.from_json(ann["annotation"], temp_meta)