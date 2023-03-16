import torch
import supervisely as sly
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin

import src.sly_globals as g


def apply_model(images, target_sizes, model, processor, query_image=None, text_queries=None, confidence_threshhold=0.5, nms_threshhold=0.5):
    if query_image is not None:
        inputs = processor(
            images=images, query_images=query_image, return_tensors="pt"
        ).to(g.DEVICE)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        outputs.logits = outputs.logits.cpu()
        outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

        results = processor.post_process_image_guided_detection(
            outputs=outputs,
            threshold=confidence_threshhold,
            nms_threshold=nms_threshhold,
            target_sizes=target_sizes,
        )
    else:
        inputs = processor(text=text_queries, images=images, return_tensors="pt").to(g.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process(
            outputs=outputs,
            target_sizes=target_sizes,
            # threshold=confidence_threshhold,
            # nms_threshold=nms_threshhold,
        )
    return results

def predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold):
    new_annotation = sly.Annotation(img_size=(image_info.height, image_info.width))
    for score, box, label in zip(scores, boxes, labels):
        if score < confidence_threshhold:
            continue
        obj_class = sly.ObjClass(label, sly.Rectangle)
        x0, y0, x1, y1 = box
        obj_label = sly.Label(sly.Rectangle(y0, x0, y1, x1), obj_class)
        new_annotation = new_annotation.add_label(obj_label)
    return new_annotation