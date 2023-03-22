import torch
import supervisely as sly
from supervisely.imaging.color import random_rgb
import src.sly_globals as g
from torchvision.ops import nms

OBJECT_COLORS = {}

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
            nms_threshold=1,
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

def predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold: float = 0.5, nms_threshhold: float=0.5):
    non_suppressed_indexes = nms(torch.Tensor(boxes), torch.Tensor(scores), nms_threshhold)
    new_annotation = sly.Annotation(img_size=(image_info.height, image_info.width))
    for i, (score, box, label) in enumerate(zip(scores, boxes, labels)):
        if score < confidence_threshhold or i not in non_suppressed_indexes:
            continue
        color = OBJECT_COLORS.get(label, random_rgb())
        if label not in OBJECT_COLORS.keys():
            OBJECT_COLORS[label] = color
        obj_class = sly.ObjClass(label, sly.Rectangle, color=color)
        x0, y0, x1, y1 = box
        obj_label = sly.Label(sly.Rectangle(y0, x0, y1, x1), obj_class)
        new_annotation = new_annotation.add_label(obj_label)
    return new_annotation

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