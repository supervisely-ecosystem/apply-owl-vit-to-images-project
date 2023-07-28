import supervisely as sly


def inference_json_anno_preprocessing(
    ann, project_meta: sly.ProjectMeta
) -> sly.Annotation:
    temp_meta = project_meta.clone()
    pred_classes = []
    for i, obj in enumerate(ann["annotation"]["objects"]):
        class_ = obj["classTitle"] + "_pred"
        pred_classes.append(class_)
        ann["annotation"]["objects"][i]["classTitle"] = class_

        if temp_meta.get_obj_class(class_) is None:
            new_obj_class = sly.ObjClass(class_, sly.Rectangle)
            temp_meta = temp_meta.add_obj_class(new_obj_class)
    if temp_meta.get_tag_meta("confidence") is None:
        temp_meta = temp_meta.add_tag_meta(
            sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        )
    return sly.Annotation.from_json(ann["annotation"], temp_meta)
