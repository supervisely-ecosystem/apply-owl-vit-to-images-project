import os
import random
from typing import List

import torch
import numpy as np
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Progress,
    Text,
    Empty,
    Container,
    Checkbox,
    Stepper,
    ClassesTable,
    ImageRegionSelector,
    ProjectThumbnail,
    RadioTabs,
    Input,
    GridGallery,
    InputNumber,
    Field,
    Progress,
    SelectAppSession,
    DoneLabel,
    ModelInfo,
    Switch,
    RadioGroup
)
import src.sly_globals as g

from src.model import apply_model, predictions_to_anno
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin

IS_LOCAL_INFERENCE = True
IS_IMAGE_PROMPT = True
PREVIEW_IMAGES_INFOS = []
CURRENT_REF_IMAGE_INDEX = 0
REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
model_data = {}

# fetching some images for preview
datasets_list = g.api.dataset.get_list(g.project_id)
image_info_list = []
for dataset in datasets_list:
    samples_count = (
        dataset.images_count
        if len(datasets_list) == 1
        else dataset.count * (100 - len(datasets_list)) / 100
    )
    image_info_list += random.sample(g.api.image.get_list(dataset.id), samples_count)
    if len(image_info_list) >= 1000:
        break
ref_image_info = image_info_list[CURRENT_REF_IMAGE_INDEX]


def get_image_path(image_name: str) -> str:
    for dataset in g.project_fs.datasets:
        if dataset.item_exists(image_name):
            return dataset.get_img_path(image_name)


######################
### Input project card
######################
project_preview = ProjectThumbnail(g.project_info)
progress_bar_download_data = Progress(hide_on_finish=False)
progress_bar_download_data.hide()
text_download_data = Text("Data has been successfully downloaded", status="success")
text_download_data.hide()
button_download_data = Button("Download")

@button_download_data.click
def download_data():
    try:
        if sly.fs.dir_exists(g.project_dir):
            sly.logger.info("Data already downloaded.")
        else:
            button_download_data.hide()
            progress_bar_download_data.show()
            sly.fs.mkdir(g.project_dir)
            with progress_bar_download_data(
                message=f"Processing images...", total=g.project_info.items_count
            ) as pbar:
                sly.Project.download(
                    api=g.api,
                    project_id=g.project_id,
                    dest_dir=g.project_dir,
                    batch_size=100,
                    progress_cb=pbar.update,
                    only_image_tags=False,
                    save_image_info=True,
                )
            sly.logger.info("Data successfully downloaded.")
        g.project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        progress_bar_download_data.hide()
        button_download_data.hide()
        text_download_data.show()
        stepper.set_active_step(2)
    except Exception as e:
        sly.logger.info("Something went wrong.")
        progress_bar_download_data.hide()
        button_download_data.show()
        text_download_data.set("Data download failed", status="error")
        text_download_data.show()
        stepper.set_active_step(1)
data_card = Card(
    title="Input data",
    content=Container(
        [
            project_preview,
            progress_bar_download_data,
            text_download_data,
            button_download_data,
        ]
    ),
)


############################
### Inference type selection
############################
confidence_threshhold_input = InputNumber(value=0.5, min=00.1, max=1, step=0.01)
nms_threshhold_input = InputNumber(value=1, min=0.01, max=1, step=0.01)
field_confidence_threshhold = Field(
    title="Confidence threshold",
    description="Threshold for the minimum confidence that a detection must have to be displayed (higher values mean fewer boxes will be shown):",
    content=confidence_threshhold_input,
)
field_nms_threshhold = Field(
    title="NMS threshold",
    description="Threshold for non-maximum suppression of overlapping boxes (higher values mean more boxes will be shown)",
    content=nms_threshhold_input,
)
select_model = SelectAppSession(
    team_id=g.team.id, tags=["deployed_owl_vit_object_detection"]
)
connect_model_done = DoneLabel("Model successfully connected.")
connect_model_done.hide()
model_info = ModelInfo()
set_inference_type_button = Button(
    text="Connect to model",
    icon="zmdi zmdi-input-composite",
    button_type="success",
)
inference_type_selection_tabs = RadioTabs(
    titles=["Local inference", "Served model"],
    contents=[
        Container(
            [field_confidence_threshhold, field_nms_threshhold],
            direction="horizontal",
        ),
        Container(
            [select_model, set_inference_type_button, connect_model_done, model_info]
        ),
    ],
    descriptions=[
        "Run model locally on the available accelerator",
        "Select served model",
    ],
)
@inference_type_selection_tabs.value_changed
def inference_type_changed(val):
    global IS_LOCAL_INFERENCE
    IS_LOCAL_INFERENCE = True if val == 'Local inference' else False
inference_type_selection_card = Card(
    title="Model settings",
    description="Select served Owl-ViT model or run it as-is on the available accelerator",
    content=inference_type_selection_tabs,
)

@set_inference_type_button.click
def connect_to_model():
    model_session_id = select_model.get_selected_id()
    if model_session_id is not None:
        set_inference_type_button.hide()
        connect_model_done.show()
        select_model.disable()
        # show model info
        model_info.set_session_id(session_id=model_session_id)
        model_info.show()
        set_inference_type_button.text = "Change model"
        # get model meta
        model_meta_json = g.api.task.send_request(
            model_session_id,
            "get_output_classes_and_tags",
            data={},
        )
        sly.logger.info(f"Model meta: {str(model_meta_json)}")
        model_data["model_meta"] = sly.ProjectMeta.from_json(model_meta_json)
        model_data["session_id"] = model_session_id


#############################
### Model input configuration
#############################
text_prompt_textarea = Input(
    placeholder="Description of object, that you want to detect via NN model"
)
class_input = Input(placeholder="The class name for selected object")
class_input_field = Field(
    content=class_input,
    title="Class name",
    description="All detected objects will be added to project/dataset with this class name",
)
image_region_selector = ImageRegionSelector(
    image_info=ref_image_info, widget_width="500px", widget_height="500px"
)
@image_region_selector.bbox_changed
def bbox_updated(new_scaled_bbox):
    sly.logger.info(f"new_scaled_bbox: {new_scaled_bbox}")

previous_image_button = Button(
    "Previous image", icon="zmdi zmdi-skip-previous", button_size="small"
)
next_image_button = Button(
    "Next image", icon="zmdi zmdi-skip-next", button_size="small"
)
random_image_button = Button(
    "New random image", icon="zmdi zmdi-refresh", button_size="small"
)
set_input_button = Button("Set model input")
previous_image_button.disable()

@previous_image_button.click
def previous_image():
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    CURRENT_REF_IMAGE_INDEX = REF_IMAGE_HISTORY[
        -(REF_IMAGE_HISTORY[::-1].index(CURRENT_REF_IMAGE_INDEX) + 2)
    ]
    image_region_selector.image_update(image_info_list[CURRENT_REF_IMAGE_INDEX])
    if CURRENT_REF_IMAGE_INDEX == REF_IMAGE_HISTORY[0]:
        previous_image_button.disable()

@next_image_button.click
def next_image():
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    if CURRENT_REF_IMAGE_INDEX != REF_IMAGE_HISTORY[-1]:
        CURRENT_REF_IMAGE_INDEX = REF_IMAGE_HISTORY[
            REF_IMAGE_HISTORY.index(CURRENT_REF_IMAGE_INDEX) + 1
        ]
    else:
        CURRENT_REF_IMAGE_INDEX += 1
        REF_IMAGE_HISTORY.append(CURRENT_REF_IMAGE_INDEX)
    REF_IMAGE_HISTORY = REF_IMAGE_HISTORY[-10:]
    image_region_selector.image_update(image_info_list[CURRENT_REF_IMAGE_INDEX])
    previous_image_button.enable()

@random_image_button.click
def random_image():
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    CURRENT_REF_IMAGE_INDEX = random.randint(0, len(image_info_list) - 1)
    REF_IMAGE_HISTORY.append(CURRENT_REF_IMAGE_INDEX)
    REF_IMAGE_HISTORY = REF_IMAGE_HISTORY[-10:]
    image_region_selector.image_update(image_info_list[CURRENT_REF_IMAGE_INDEX])
    previous_image_button.enable()
    next_image_button.enable()

@set_input_button.click
def set_model_input():
    if model_settings_card.is_disabled() is False:
        model_settings_card.disable()
        model_input_tabs.disable()
        previous_image_button.disable()
        next_image_button.disable()
        random_image_button.disable()
        image_region_selector.disable()
        text_prompt_textarea.disable()
        set_input_button.text = "Change model input"
    else:
        model_settings_card.enable()
        model_input_tabs.enable()
        previous_image_button.enable()
        next_image_button.enable()
        random_image_button.enable()
        image_region_selector.enable()
        text_prompt_textarea.enable()
        set_input_button.text = "Set model input"

model_input_tabs = RadioTabs(
    titles=["Reference image", "Text prompt"],
    contents=[
        Container(
            [
                Container(
                    [previous_image_button, next_image_button, random_image_button],
                    direction="horizontal",
                ),
                image_region_selector,
                class_input_field,
            ]
        ),
        Container([text_prompt_textarea]),
    ],
    descriptions=[
        "Pick object by bounding box editing",
        "Describe object, that you want to detect",
    ],
)
@model_input_tabs.value_changed
def model_input_changed(val):
    global IS_IMAGE_PROMPT
    IS_IMAGE_PROMPT = True if val == 'Reference image' else False

model_settings_card = Card(
    title="Model input configuration",
    description="Configure input for model as text-prompt or as reference image",
    content=Container([model_input_tabs, set_input_button]),
)


###################
### Results preview
###################
grid_gallery = GridGallery(
    columns_number=g.COLUMNS_COUNT,
    annotations_opacity=0.5,
    show_opacity_slider=True,
    enable_zoom=False,
    sync_views=False,
    fill_rectangle=True,
    show_preview=True,
)
update_images_preview_button = Button("New random images", icon="zmdi zmdi-refresh")
@update_images_preview_button.click
def update_images_preview():
    grid_gallery.clean_up()
    NEW_PREVIEW_IMAGES_INFOS = []
    for i in range(g.PREVIEW_IMAGES_COUNT):
        img_info = random.choice(image_info_list)
        NEW_PREVIEW_IMAGES_INFOS.append(img_info)
        grid_gallery.append(
            title=img_info.name,
            image_url=img_info.preview_url,
            column_index=int(i % g.COLUMNS_COUNT),
        )
    global PREVIEW_IMAGES_INFOS
    PREVIEW_IMAGES_INFOS = NEW_PREVIEW_IMAGES_INFOS
update_images_preview()

update_predictions_preview_button = Button(
    "Predictions preview", icon="zmdi zmdi-labels"
)
@update_predictions_preview_button.click
def update_predictions_preview():
    global IS_IMAGE_PROMPT
    confidence_threshhold = confidence_threshhold_input.get_value()
    nms_threshhold = nms_threshhold_input.get_value()

    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = model.to(g.DEVICE)
    model.eval()

    annotations_list = []
    for i, image_info in enumerate(PREVIEW_IMAGES_INFOS):
        image = sly.image.read(get_image_path(image_info.name))
        target_sizes = torch.Tensor([[image_info.height, image_info.width]]).to(g.DEVICE)

        if IS_IMAGE_PROMPT:
            selected_bbox = image_region_selector.scaled_bbox
            x0, y0, x1, y1 = np.array(selected_bbox).reshape(-1)
            query_image = sly.image.read(get_image_path(ref_image_info.name))
            query_image = query_image[y0:y1, x0:x1]
            results = apply_model(
                image, 
                target_sizes, 
                model, 
                processor, 
                query_image, 
                confidence_threshhold=confidence_threshhold, 
                nms_threshhold=nms_threshhold
            )
        else:
            text_queries = text_prompt_textarea.get_value().split(";")
            results = apply_model(
                image, 
                target_sizes, 
                model, 
                processor, 
                text_queries=text_queries, 
                confidence_threshhold=confidence_threshhold, 
                nms_threshhold=nms_threshhold
            )


        scores = results[0]["scores"].cpu().detach().numpy()
        boxes = results[0]["boxes"].cpu().detach().numpy()
        labels = results[0]["labels"]
        if labels is None:
            labels = [class_input.get_value()] * len(boxes)
        else:
            labels = [text_queries[label] for label in labels]
        new_annotation = predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold)
        annotations_list.append(new_annotation)
        sly.logger.info(
            f"{i+1} image processed. {len(PREVIEW_IMAGES_INFOS) - (i+1)} images left."
        )

    grid_gallery.clean_up()
    for i, (image_info, annotation) in enumerate(
        zip(PREVIEW_IMAGES_INFOS, annotations_list)
    ):
        grid_gallery.append(
            image_url=image_info.preview_url,
            annotation=annotation,
            title=image_info.name,
            column_index=int(i % g.COLUMNS_COUNT),
        )

preview_card = Card(
    title="Preview results",
    description="Model prediction result preview",
    content=grid_gallery,
    content_top_right=Container(
        [update_images_preview_button, update_predictions_preview_button],
        direction="horizontal",
    ),
)

#######################
### Applying model card
#######################
keep_existed_annotations = Checkbox('Save existed project annotations')
output_project_name_input = Input(value=f"{g.project_info.name} - (Annotated)")
output_project_name_field = Field(output_project_name_input, "Output project name")
apply_progress_bar = Progress(hide_on_finish=False)
select_output_destination = RadioGroup(
    items=[
        RadioGroup.Item(value="Save annotations to new project"),
        RadioGroup.Item(value="Add annotations to selected project"),
    ],
    direction="vertical",
)
@select_output_destination.value_changed
def select_output_destination_changed(format):
    if format == "Save annotations to new project":
        output_project_name_field.show()
    else:
        output_project_name_field.hide()
run_model_button = Button("Run model")


@run_model_button.click
def run_model():
    global IS_LOCAL_INFERENCE
    confidence_threshhold = confidence_threshhold_input.get_value()
    nms_threshhold = nms_threshhold_input.get_value()

    output_destination = select_output_destination.get_value() 
    if output_destination == "Save annotations to new project":
        is_new_project = True
        output_project_id = None
    else:
        is_new_project = False
        output_project_id = g.project_id

    is_new_project = True if output_project_id is None else False
    if output_project_id is None:
        output_project_name = output_project_name_input.get_value()
        if output_project_name.strip() == '':
            output_project_name = f"{g.project_info.name} - (Annotated)"
        output_project = g.api.project.create(
            workspace_id=g.workspace.id,
            name=output_project_name,
            type=sly.ProjectType.IMAGES,
            change_name_if_conflict=True,
        )
        output_project_id = output_project.id
        output_project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(output_project_id))
        output_project_meta = output_project_meta.merge(g.project_meta)

    if IS_LOCAL_INFERENCE:
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = model.to(g.DEVICE)
        model.eval()

    # apply models to project
    with apply_progress_bar(message="Applying model to project...", total=g.project_info.images_count) as pbar:
        for dataset in datasets_list:
            # annotate image in its dataset
            if is_new_project is True:
                output_dataset = g.api.dataset.create(
                    project_id=output_project_id, name=dataset.name, change_name_if_conflict=True
                )
            else:
                output_dataset = dataset
                
            images_info = g.api.image.get_list(dataset.id)
            for image_info in images_info:
                if keep_existed_annotations.is_checked():
                    image_ann_json = g.api.annotation.download(image_info.id).annotation
                    image_ann = sly.Annotation.from_json(image_ann_json, output_project_meta)
                else:
                    image_ann = sly.Annotation(img_size=(image_info.height, image_info.width))
                
                text_queries = text_prompt_textarea.get_value().split(";")
                selected_bbox = image_region_selector.scaled_bbox
                x0, y0, x1, y1 = np.array(selected_bbox).reshape(-1)
                
                if IS_LOCAL_INFERENCE:
                    image = sly.image.read(get_image_path(image_info.name))
                    target_sizes = torch.Tensor([[image_info.height, image_info.width]]).to(g.DEVICE)

                    if IS_IMAGE_PROMPT:
                        query_image = sly.image.read(get_image_path(ref_image_info.name))
                        query_image = query_image[y0:y1, x0:x1]
                        results = apply_model(
                            image, 
                            target_sizes, 
                            model, 
                            processor, 
                            query_image, 
                            confidence_threshhold=confidence_threshhold, 
                            nms_threshhold=nms_threshhold
                        )
                    else:
                        results = apply_model(
                            image, 
                            target_sizes, 
                            model, 
                            processor, 
                            text_queries=text_queries, 
                            confidence_threshhold=confidence_threshhold, 
                            nms_threshhold=nms_threshhold
                        )
                    scores = results[0]["scores"].cpu().detach().numpy()
                    boxes = results[0]["boxes"].cpu().detach().numpy()
                    labels = results[0]["labels"]
                    if labels is None:
                        labels = [class_input.get_value()] * len(boxes)
                    else:
                        labels = [text_queries[label] for label in labels]
                    ann = predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold)
                else:
                    ann = g.api.task.send_request(
                        model_data["session_id"],
                        "inference_image_id",
                        data={"image_id": image_info.id, "settings": inference_settings},
                        timeout=500,
                    )
                    ann = sly.Annotation.from_json(ann["annotation"], output_project_meta)

                for target_class_name in set(labels):
                    target_class = output_project_meta.get_obj_class(target_class_name)
                    if target_class is None:  # if obj class is not in output project meta
                        target_class = sly.ObjClass(target_class_name, sly.Rectangle)
                        output_project_meta = output_project_meta.add_obj_class(target_class)
                
                image_ann = image_ann.add_labels(ann.labels)
                g.api.project.update_meta(output_project_id, output_project_meta.to_json())
                new_image_info = g.api.image.copy(dst_dataset_id=output_dataset.id, id=image_info.id, change_name_if_conflict=True, with_annotations=False)
                g.api.annotation.upload_ann(new_image_info.id, image_ann)
                pbar.update()

    output_project.set_meta(output_project_meta)
    output_project_thmb.set(info=output_project)

    output_project_thmb.show()
    sly.logger.info("Project was successfully labeled")
    app.shutdown()


output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
run_model_card = Card(
    title="Apply model",
    content=Container([select_output_destination, output_project_name_field, keep_existed_annotations, run_model_button, apply_progress_bar, output_project_thmb]),
)

stepper = Stepper(
    widgets=[
        data_card,
        inference_type_selection_card,
        model_settings_card,
        preview_card,
        run_model_card,
    ]
)
app = sly.Application(layout=stepper)
