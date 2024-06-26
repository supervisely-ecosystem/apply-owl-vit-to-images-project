import supervisely as sly
import src.sly_globals as g
from typing import List
from supervisely.app.widgets import Progress


apply_progress_bar = Progress(hide_on_finish=False)


def run(
    destination_project: sly.app.widgets.DestinationProject,
    inference_settings,
    MODEL_DATA,
):
    def add_new_classes_to_proj_meta(
        anns: List[sly.Annotation], output_project_meta: sly.ProjectMeta
    ) -> sly.ProjectMeta:
        project_meta_needs_update = False
        for ann in anns:
            for label in ann.labels:
                label.obj_class.name += "_pred"
                if output_project_meta.get_obj_class(label) is None:
                    new_obj_class = sly.ObjClass(label, sly.Rectangle)
                    output_project_meta = output_project_meta.add_obj_class(new_obj_class)
                    project_meta_needs_update = True

        if project_meta_needs_update:
            g.api.project.update_meta(output_project_id, output_project_meta)
        return output_project_meta

    output_project_id = destination_project.get_selected_project_id()
    if output_project_id is None:
        output_project_name = destination_project.get_project_name()
        if output_project_name.strip() == "":
            output_project_name = f"{g.project_info.name} - (Annotated)"
        output_project = g.api.project.create(
            workspace_id=g.workspace.id,
            name=output_project_name,
            type=sly.ProjectType.IMAGES,
            change_name_if_conflict=True,
        )
        output_project_id = output_project.id

    def get_output_ds(destination_project: sly.app.widgets.DestinationProject, dataset_name):
        use_project_datasets_structure = destination_project.use_project_datasets_structure()
        if use_project_datasets_structure is True:
            output_dataset_name = dataset_name
        else:
            output_dataset_id = destination_project.get_selected_dataset_id()
            if not output_dataset_id:
                output_dataset_name = destination_project.get_dataset_name()
                if not output_dataset_name or output_dataset_name.strip() == "":
                    output_dataset_name = "ds"
        output_dataset = g.api.dataset.get_or_create(output_project.id, output_dataset_name)

        return output_dataset.id

    # merge project metas and add tag "confidence" to it
    output_project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(output_project_id))
    output_project_meta = output_project_meta.merge(g.project_meta)
    if output_project_meta.get_tag_meta("confidence") is None:
        output_project_meta = output_project_meta.add_tag_meta(
            sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        )
        g.api.project.update_meta(output_project_id, output_project_meta)

    # apply models to project
    datasets_list = [g.api.dataset.get_info_by_id(ds_id) for ds_id in g.DATASET_IDS]
    total_items_cnt = sum([ds.items_count for ds in datasets_list])
    with apply_progress_bar(message="Applying model to project...", total=total_items_cnt) as pbar:
        for dataset in datasets_list:
            images_info = g.api.image.get_list(dataset.id)
            output_dataset_id = get_output_ds(destination_project, dataset.name)
            for img_infos_batch in sly.batched(images_info):
                img_ids = [image_info.id for image_info in img_infos_batch]

                # get existing and new annotations
                image_anns = [
                    sly.Annotation.from_json(img_ann.annotation, output_project_meta)
                    for img_ann in g.api.annotation.download_batch(dataset.id, img_ids)
                ]
                new_anns = [
                    g.api.task.send_request(
                        MODEL_DATA["session_id"],
                        "inference_image_id",
                        data={
                            "image_id": image_info.id,
                            "settings": inference_settings,
                        },
                        timeout=500,
                    )
                    for image_info in img_infos_batch
                ]

                output_project_meta = add_new_classes_to_proj_meta(new_anns, output_project_meta)

                # merge annotations
                result_anns = []
                for image_ann, new_ann in zip(image_anns, new_anns):
                    new_ann = sly.Annotation.from_json(new_ann["annotation"], output_project_meta)
                    sly.logger.debug(f"New annotations added to images: {len(new_ann.labels)}")
                    result_anns.append(image_ann.add_labels(new_ann.labels))

                # upload new data
                # if (
                #     destination_project.get_selected_project_id() != g.project_id
                #     or destination_project.get_selected_dataset_id() != g.dataset_id
                # ):
                image_names = [image_info.name for image_info in img_infos_batch]
                image_infos = g.api.image.upload_ids(
                    output_dataset_id,
                    image_names,
                    img_ids,
                    conflict_resolution=destination_project.get_conflict_resolution(),
                )
                img_ids = [image_info.id for image_info in image_infos]
                g.api.annotation.upload_anns(img_ids, result_anns)
                pbar.update(len(img_ids))

    return g.api.project.get_info_by_id(output_project_id)
