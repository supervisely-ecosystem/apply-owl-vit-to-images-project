import os
import random
import shutil
from typing import List

import torch
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Progress,
    Container,
    Checkbox,
    Stepper,
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
    RadioGroup,
    RadioTable,
    SelectDataset,
)
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import src.sly_globals as g
from src.model import apply_model, predictions_to_anno, inference_json_anno_preprocessing

IS_LOCAL_INFERENCE = True
IS_IMAGE_PROMPT = True
PREVIEW_IMAGES_INFOS = []
CURRENT_REF_IMAGE_INDEX = 0
REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
MODEL_DATA = {}

# fetching some images for preview
def get_images_infos_for_preview():
    if len(g.DATASET_IDS) > 0:
        datasets_list = [g.api.dataset.get_info_by_id(ds_id) for ds_id in g.DATASET_IDS]
    else:
        datasets_list = g.api.dataset.get_list(g.project_id)
    IMAGES_INFO_LIST = []
    for dataset in datasets_list:
        samples_count = (
            dataset.images_count
            if len(datasets_list) == 1
            else dataset.images_count * (100 - len(datasets_list)) // 100
        )
        IMAGES_INFO_LIST += random.sample(g.api.image.get_list(dataset.id), samples_count)
        if len(IMAGES_INFO_LIST) >= 1000:
            break
    return IMAGES_INFO_LIST
IMAGES_INFO_LIST = get_images_infos_for_preview()


def get_image_path(image_name: str) -> str:
    for dataset in g.project_fs.datasets:
        if dataset.item_exists(image_name):
            return dataset.get_img_path(image_name)

######################
### Input project card
######################
dataset_selector = SelectDataset(project_id=g.project_id, multiselect=True, select_all_datasets=True)
text_download_data = DoneLabel("Data was successfully downloaded.")
text_download_data.hide()
button_download_data = Button("Select data")
progress_bar_download_data = Progress(hide_on_finish=False)
progress_bar_download_data.hide()
data_card = Card(
    title="Input data selection", 
    content=Container([
        dataset_selector,
        progress_bar_download_data,
        text_download_data,
        button_download_data,
    ])
)
@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    global IMAGES_INFO_LIST, CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    button_download_data.loading = True
    new_project_id = dataset_selector._project_selector.get_selected_id()
    if new_project_id != g.project_id:
        dataset_selector.disable()
        g.project_id = new_project_id
        g.project_info = g.api.project.get_info_by_id(g.project_id)
        g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
        g.workspace = g.api.workspace.get_info_by_id(g.project_info.workspace_id)
        output_project_name_input.set_value(f"{g.project_info.name} - (Annotated)")
        dataset_selector.enable()
        g.DATASET_IDS = new_dataset_ids
        IMAGES_INFO_LIST = get_images_infos_for_preview()
        update_images_preview()
        CURRENT_REF_IMAGE_INDEX = 0
        REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
        image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])
    else:
        if set(g.DATASET_IDS) != set(new_dataset_ids):
            g.DATASET_IDS = new_dataset_ids
            text_download_data.hide()
            IMAGES_INFO_LIST = get_images_infos_for_preview()
            update_images_preview()
            CURRENT_REF_IMAGE_INDEX = 0
            REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
            image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])

    sly.logger.info(f"Team: {g.team.id} \t Project: {g.project_info.id} \t Datasets: {g.DATASET_IDS}")
    
    if len(new_dataset_ids) == 0:
        button_download_data.hide()
    else:
        button_download_data.show()
    button_download_data.loading = False


@button_download_data.click
def download_data():
    if data_card.is_disabled() is True:
        toggle_cards(['data_card'], enabled=True)
        toggle_cards(['inference_type_selection_card', 'model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
        button_download_data.enable()
        text_download_data.hide()
        button_download_data.text = "Select data"
        set_model_type_button.disable()
        set_model_type_button.text = 'Select model' 
        set_input_button.disable()
        set_input_button.text = 'Set model input' 
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        run_model_button.disable()
        stepper.set_active_step(1)
    else:
        button_download_data.disable()
        toggle_cards(['data_card', 'inference_type_selection_card', 'model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
        g.project_dir = os.path.join(g.projects_dir, str(g.project_id))
        try:
            if sly.fs.dir_exists(g.project_dir):
                tmp_project = sly.Project(g.project_dir, sly.OpenMode.READ)
                selected_datasets = [g.api.dataset.get_info_by_id(id) for id in g.DATASET_IDS]
                missed_datasets = [ds for ds in selected_datasets if ds.name not in tmp_project.datasets.keys()]
                missed_datasets_ids = [ds.id for ds in missed_datasets]
                missed_items_cnt = sum([ds.items_count for ds in missed_datasets])
                if len(missed_datasets) > 0:
                    sly.logger.info(f"Datasets {missed_datasets_ids} were missed in project directory: {g.project_dir}.")
                    temp_project_dir = os.path.join(g.projects_dir, 'temp_project')

                    progress_bar_download_data.show()
                    with progress_bar_download_data(
                        message=f"Processing images...", total=missed_items_cnt
                    ) as pbar:
                        sly.Project.download(
                            api=g.api,
                            project_id=g.project_id,
                            dest_dir=temp_project_dir,
                            batch_size=100,
                            dataset_ids=missed_datasets_ids,
                            progress_cb=pbar.update,
                            only_image_tags=False,
                            save_image_info=True,
                        )
                    for ds in missed_datasets:
                        src_ds_path = os.path.join(temp_project_dir, ds.name)
                        dst_ds_path = os.path.join(g.project_dir, ds.name)
                        shutil.move(src_ds_path, dst_ds_path)
                    sly.fs.remove_dir(temp_project_dir)
                    sly.logger.info("Missed datasets was succesfully downloaded.")
                else:
                    sly.logger.info("Data already downloaded.")
            else:
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
                        dataset_ids=g.DATASET_IDS,
                        progress_cb=pbar.update,
                        only_image_tags=False,
                        save_image_info=True,
                    )
                sly.logger.info("Data successfully downloaded.")
            g.project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
            button_download_data.text = 'Change data' 
            text_download_data.show()
            toggle_cards(['inference_type_selection_card'], enabled=True)
            set_model_type_button.enable()
            stepper.set_active_step(2)
        except Exception as e:
            sly.logger.info("Something went wrong.")
            button_download_data.text = 'Select data' 
            text_download_data.set("Data download failed", status="error")
            text_download_data.show()
            toggle_cards(['inference_type_selection_card', 'model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
            set_model_type_button.disable()
            stepper.set_active_step(1)
        finally:
            button_download_data.enable()
            progress_bar_download_data.hide()

############################
### Inference type selection
############################
select_model_session = SelectAppSession(
    team_id=g.team.id, tags=["deployed_owl_vit_object_detection"]
)
@select_model_session.value_changed
def select_model_session_change(val):
    if val is None:
        if IS_LOCAL_INFERENCE is False:
            set_model_type_button.disable()
    else:
        set_model_type_button.enable()

connect_model_done = DoneLabel("Model successfully connected.")
connect_model_done.hide()
model_info = ModelInfo()
set_model_type_button = Button(text="Select model")
model_config_table = RadioTable(
    columns=["Model", "Backbone", "Pretraining", "Size", "LVIS AP", "LVIS APr", ], 
    rows=[
        ["OWL-ViT base patch 32", "ViT-B/32","CLIP", "583 MB", "19.3", "16.9",],
        ["OWL-ViT base patch 16", "ViT-B/16","CLIP", "581 MB", "20.8", "17.1",],
        ["OWL-ViT large patch 14", "ViT-L/14","CLIP", "1.65 GB", "34.6", "31.2",],
    ], 
)
inference_type_selection_tabs = RadioTabs(
    titles=["Local inference", "Served model"],
    contents=[
        model_config_table,
        Container(
            [select_model_session, connect_model_done, model_info]
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
    if val == 'Local inference':
        IS_LOCAL_INFERENCE = True 
        set_model_type_button.text = 'Select model'
        set_model_type_button.enable()
    else:
        IS_LOCAL_INFERENCE = False
        set_model_type_button.text = 'Connect to model'
        if select_model_session.get_selected_id() is None:
            set_model_type_button.disable()

inference_type_selection_card = Card(
    title="Model settings",
    description="Select served Owl-ViT model or run it as-is on the available accelerator",
    content=Container([inference_type_selection_tabs, set_model_type_button]),
)

@set_model_type_button.click
def set_model_type():
    global IS_LOCAL_INFERENCE, MODEL_DATA

    if IS_LOCAL_INFERENCE:
        if set_model_type_button.text == 'Change model':
            MODEL_DATA = {}
            model_config_table.enable()
            inference_type_selection_tabs.enable()
            set_model_type_button.text = 'Select model'
            toggle_cards(['model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
            set_input_button.disable()
            set_input_button.text = 'Set model input'
            update_images_preview_button.disable()
            update_predictions_preview_button.disable()
            run_model_button.disable()
            stepper.set_active_step(2)
        else:
            inference_type_selection_tabs.disable()
            MODEL_DATA = dict(zip(map(str.lower, model_config_table.columns), model_config_table.get_selected_row()))
            model_config_table.disable()
            set_model_type_button.text = 'Change model'
            toggle_cards(['model_settings_card'], enabled=True)
            set_input_button.enable()
            stepper.set_active_step(3)
    else:
        if set_model_type_button.text == 'Disconnect model':
            MODEL_DATA["session_id"] = None
            MODEL_DATA["model_meta"] = None
            connect_model_done.hide()
            model_info.hide()
            select_model_session.enable()
            inference_type_selection_tabs.enable()
            set_model_type_button.enable()
            set_model_type_button.text = 'Connect to model'
            toggle_cards(['model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
            set_input_button.disable()
            set_input_button.text = 'Set model input'
            update_images_preview_button.disable()
            update_predictions_preview_button.disable()
            run_model_button.disable()
            stepper.set_active_step(2)
        else:
            model_session_id = select_model_session.get_selected_id()
            if model_session_id is not None:
                try:
                    set_model_type_button.disable()
                    inference_type_selection_tabs.disable()
                    select_model_session.disable()
                    # get model meta
                    model_meta_json = g.api.task.send_request(
                        model_session_id,
                        "get_output_classes_and_tags",
                        data={},
                    )
                    sly.logger.info(f"Model meta: {str(model_meta_json)}")
                    MODEL_DATA["model_meta"] = sly.ProjectMeta.from_json(model_meta_json)
                    MODEL_DATA["session_id"] = model_session_id
                    connect_model_done.show()
                    model_info.set_session_id(session_id=model_session_id)
                    model_info.show()
                    set_model_type_button.text = 'Disconnect model'
                    set_model_type_button._plain = True
                    set_model_type_button.enable()
                    toggle_cards(['model_settings_card'], enabled=True)
                    set_input_button.enable()
                    stepper.set_active_step(3)
                except Exception as e:
                    sly.logger.error(f"Cannot to connect to model. {e}")
                    set_model_type_button.enable()
                    set_model_type_button.text = 'Connect to model'
                    inference_type_selection_tabs.enable()
                    connect_model_done.hide()
                    select_model_session.enable()
                    model_info.hide()
                    toggle_cards(['model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
                    set_input_button.disable()
                    update_images_preview_button.disable()
                    update_predictions_preview_button.disable()
                    run_model_button.disable()
                    stepper.set_active_step(2)


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
image_region_selector = ImageRegionSelector(widget_width="500px", widget_height="500px")
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
    image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])
    if CURRENT_REF_IMAGE_INDEX == REF_IMAGE_HISTORY[0]:
        previous_image_button.disable()
    next_image_button.enable()

@next_image_button.click
def next_image():
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    if CURRENT_REF_IMAGE_INDEX != REF_IMAGE_HISTORY[-1]:
        CURRENT_REF_IMAGE_INDEX = REF_IMAGE_HISTORY[
            REF_IMAGE_HISTORY.index(CURRENT_REF_IMAGE_INDEX) + 1
        ]
    else:
        if CURRENT_REF_IMAGE_INDEX < len(IMAGES_INFO_LIST) - 1:
            CURRENT_REF_IMAGE_INDEX += 1
            REF_IMAGE_HISTORY.append(CURRENT_REF_IMAGE_INDEX)

    if len(IMAGES_INFO_LIST) - 1 == CURRENT_REF_IMAGE_INDEX:
        next_image_button.disable()
    REF_IMAGE_HISTORY = REF_IMAGE_HISTORY[-10:]
    image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])
    previous_image_button.enable()

@random_image_button.click
def random_image():
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    CURRENT_REF_IMAGE_INDEX = random.randint(0, len(IMAGES_INFO_LIST) - 1)
    REF_IMAGE_HISTORY.append(CURRENT_REF_IMAGE_INDEX)
    REF_IMAGE_HISTORY = REF_IMAGE_HISTORY[-10:]
    image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])
    if CURRENT_REF_IMAGE_INDEX != REF_IMAGE_HISTORY[0]:
        previous_image_button.enable()
    if CURRENT_REF_IMAGE_INDEX < len(IMAGES_INFO_LIST) - 1:
        next_image_button.enable()

@set_input_button.click
def set_model_input():
    if model_settings_card.is_disabled() is True:
        set_input_button.text = "Set model input"
        toggle_cards(['model_settings_card'], enabled=True)
        toggle_cards(['preview_card', 'run_model_card'], enabled=False)
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        run_model_button.disable()
        stepper.set_active_step(3)
    else:
        set_input_button.text = "Change model input"
        toggle_cards(['model_settings_card'], enabled=False)
        toggle_cards(['preview_card', 'run_model_card'], enabled=True)
        update_images_preview_button.enable()
        update_predictions_preview_button.enable()
        run_model_button.enable()
        stepper.set_active_step(5)

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

confidence_threshhold_input = InputNumber(value=0.5, min=00.1, max=1, step=0.01)
nms_threshhold_input = InputNumber(value=0.5, min=0.01, max=1, step=0.01)
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
model_settings_card = Card(
    title="Model input configuration",
    description="Configure input for model as text-prompt or as reference image",
    content=Container([model_input_tabs, Container([field_confidence_threshhold, field_nms_threshhold], direction="horizontal"), set_input_button]),
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
    update_predictions_preview_button.disable()
    global IMAGES_INFO_LIST

    grid_gallery.clean_up()
    NEW_PREVIEW_IMAGES_INFOS = []
    for i in range(g.PREVIEW_IMAGES_COUNT):
        img_info = random.choice(IMAGES_INFO_LIST)
        NEW_PREVIEW_IMAGES_INFOS.append(img_info)
        grid_gallery.append(
            title=img_info.name,
            image_url=img_info.preview_url,
            column_index=int(i % g.COLUMNS_COUNT),
        )
    global PREVIEW_IMAGES_INFOS
    PREVIEW_IMAGES_INFOS = NEW_PREVIEW_IMAGES_INFOS
    update_predictions_preview_button.enable()

update_predictions_preview_button = Button(
    "Predictions preview", icon="zmdi zmdi-labels"
)
@update_predictions_preview_button.click
def update_predictions_preview():
    update_images_preview_button.disable()
    global IS_LOCAL_INFERENCE, IS_IMAGE_PROMPT
    confidence_threshhold = confidence_threshhold_input.get_value()
    nms_threshhold = nms_threshhold_input.get_value()

    # for TEXT PROMPT
    text_queries = text_prompt_textarea.get_value().split(";")
    # for IMAGE REFERENCE
    selected_bbox = image_region_selector.scaled_bbox
    x0, y0, x1, y1 = *selected_bbox[0], *selected_bbox[1]

    if IS_LOCAL_INFERENCE:
        selected_model = MODEL_DATA['model']
        if selected_model == "OWL-ViT base patch 32":
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        elif selected_model == "OWL-ViT base patch 16":
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
        elif selected_model == "OWL-ViT large patch 14":
            processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        model = model.to(g.DEVICE)
        model.eval()

    annotations_list = []
    for i, image_info in enumerate(PREVIEW_IMAGES_INFOS):
        image = sly.image.read(get_image_path(image_info.name))
        target_sizes = torch.Tensor([[image_info.height, image_info.width]]).to(g.DEVICE)

        if IS_LOCAL_INFERENCE:
            if IS_IMAGE_PROMPT:
                ref_image_info = IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX]
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
                labels = [f"{class_input.get_value()}_pred"] * len(boxes)
            else:
                labels = [f"{text_queries[label]}_pred" for label in labels]
            new_annotation = predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold, nms_threshhold)
        else:
            if IS_IMAGE_PROMPT:
                inference_settings = dict(
                    mode = "reference_image",
                    reference_bbox = [y0, x0, y1, x1],
                    reference_image_id = image_region_selector.image_id,
                    reference_class_name = class_input.get_value(),
                    confidence_threshold = confidence_threshhold,
                    # nms_threshhold=nms_threshhold,
                )
                ann = g.api.task.send_request(
                    MODEL_DATA["session_id"],
                    "inference_image_id",
                    data={"image_id": image_info.id, "settings": inference_settings},
                    timeout=500,
                )
            else:
                text_queries = text_prompt_textarea.get_value().split(";")
                inference_settings = dict(
                    mode = "text_prompt",
                    text_queries = text_queries,
                    confidence_threshold = confidence_threshhold,
                    nms_threshhold=nms_threshhold,
                )
                ann = g.api.task.send_request(
                    MODEL_DATA["session_id"],
                    "inference_image_id",
                    data={"image_id": image_info.id, "settings": inference_settings},
                    timeout=500,
                )
            new_annotation = inference_json_anno_preprocessing(ann, g.project_meta)

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
    update_images_preview_button.enable()

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
keep_existed_annotations.hide()
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
        keep_existed_annotations.show()
    else:
        output_project_name_field.hide()
        keep_existed_annotations.hide()
run_model_button = Button("Run model")


@run_model_button.click
def run_model():
    try:
        toggle_cards(['data_card', 'inference_type_selection_card', 'model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
        button_download_data.disable()
        set_input_button.disable()
        next_image_button.disable()
        random_image_button.disable()
        previous_image_button.disable()
        set_model_type_button.disable()
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        output_project_thmb.hide()
        global IS_LOCAL_INFERENCE, IS_IMAGE_PROMPT, MODEL_DATA
        confidence_threshhold = confidence_threshhold_input.get_value()
        nms_threshhold = nms_threshhold_input.get_value()

        output_destination = select_output_destination.get_value() 
        if output_destination == "Save annotations to new project":
            is_new_project = True
            output_project_id = None

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
        else:
            is_new_project = False
            output_project_id = g.project_id
            output_project_meta = g.project_meta

        if IS_LOCAL_INFERENCE:
            selected_model = MODEL_DATA['model']
            if selected_model == "OWL-ViT base patch 32":
                processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
                model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            elif selected_model == "OWL-ViT base patch 16":
                processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
                model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
            elif selected_model == "OWL-ViT large patch 14":
                processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
                model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
            model = model.to(g.DEVICE)
            model.eval()

        # apply models to project
        with apply_progress_bar(message="Applying model to project...", total=g.project_info.images_count) as pbar:
            datasets_list = [g.api.dataset.get_info_by_id(ds_id) for ds_id in g.DATASET_IDS]
            for dataset in datasets_list:
                # annotate image in its dataset
                if is_new_project:
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
                    x0, y0, x1, y1 = *selected_bbox[0], *selected_bbox[1]
                    
                    if IS_LOCAL_INFERENCE:
                        image = sly.image.read(get_image_path(image_info.name))
                        target_sizes = torch.Tensor([[image_info.height, image_info.width]]).to(g.DEVICE)

                        if IS_IMAGE_PROMPT:
                            ref_image_info = IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX]
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
                            labels = [f"{class_input.get_value()}_pred"] * len(boxes)
                        else:
                            labels = [f"{text_queries[label]}_pred" for label in labels]
                        ann = predictions_to_anno(scores, boxes, labels, image_info, confidence_threshhold, nms_threshhold)
                    else:
                        if IS_IMAGE_PROMPT:
                            inference_settings = dict(
                                mode = "reference_image",
                                reference_bbox = [y0, x0, y1, x1],
                                reference_image_id = image_region_selector.image_id,
                                reference_class_name = class_input.get_value(),
                                confidence_threshold = confidence_threshhold,
                                # nms_threshhold=nms_threshhold,
                            )
                            ann = g.api.task.send_request(
                                MODEL_DATA["session_id"],
                                "inference_image_id",
                                data={"image_id": image_info.id, "settings": inference_settings},
                                timeout=500,
                            )
                        else:
                            inference_settings = dict(
                                mode = "text_prompt",
                                text_queries = text_queries,
                                confidence_threshold = confidence_threshhold,
                                nms_threshhold=nms_threshhold,
                            )
                            ann = g.api.task.send_request(
                                MODEL_DATA["session_id"],
                                "inference_image_id",
                                data={"image_id": image_info.id, "settings": inference_settings},
                                timeout=500,
                            )
                        ann = inference_json_anno_preprocessing(ann, output_project_meta)
                        labels = [label.obj_class.name for label in ann.labels]
                        if output_project_meta.get_tag_meta('confidence') is None:
                            output_project_meta = output_project_meta.add_tag_meta(sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER))

                    for target_class_name in set(labels):
                        target_class = output_project_meta.get_obj_class(target_class_name)
                        if target_class is None:  # if obj class is not in output project meta
                            target_class = sly.ObjClass(target_class_name, sly.Rectangle)
                            output_project_meta = output_project_meta.add_obj_class(target_class)
                    
                    image_ann = image_ann.add_labels(ann.labels)
                    g.api.project.update_meta(output_project_id, output_project_meta.to_json())
                    if is_new_project:
                        image_info = g.api.image.copy(dst_dataset_id=output_dataset.id, id=image_info.id, change_name_if_conflict=True, with_annotations=False)
                    g.api.annotation.upload_ann(image_info.id, image_ann)
                    pbar.update()

        output_project_info = g.api.project.get_info_by_id(output_project_id)
        output_project_thmb.set(info=output_project_info)
        output_project_thmb.show()
        sly.logger.info("Project was successfully labeled")
    except Exception as e:
        sly.logger.error('Something went wrong. Error: {e}')
    finally:
        toggle_cards(['run_model_card'], enabled=True)
        button_download_data.enable()
        run_model_button.enable()

output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
run_model_card = Card(
    title="Apply model",
    content=Container([select_output_destination, output_project_name_field, keep_existed_annotations, run_model_button, apply_progress_bar, output_project_thmb]),
)

def toggle_cards(cards: List[str], enabled: bool = False):
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY
    if 'data_card' in cards:
        if enabled:
            data_card.enable()
            dataset_selector.enable()
        else:
            data_card.disable()
            dataset_selector.disable()
    if 'inference_type_selection_card' in cards:
        if enabled:
            inference_type_selection_card.enable()
            select_model_session.enable()
            model_config_table.enable()
            inference_type_selection_tabs.enable()
        else:
            inference_type_selection_card.disable()
            select_model_session.disable()
            model_config_table.disable()
            inference_type_selection_tabs.disable()
    if 'model_settings_card' in cards:
        if enabled:
            model_settings_card.enable()
            text_prompt_textarea.enable()
            class_input.enable()
            image_region_selector.enable()
            if CURRENT_REF_IMAGE_INDEX != REF_IMAGE_HISTORY[0]:
                previous_image_button.enable()
            if CURRENT_REF_IMAGE_INDEX < len(IMAGES_INFO_LIST) - 1:
                next_image_button.enable()
            confidence_threshhold_input.enable()
            nms_threshhold_input.enable()
            random_image_button.enable()
            model_input_tabs.enable()
        else:
            model_settings_card.disable()
            text_prompt_textarea.disable()
            class_input.disable()
            image_region_selector.disable()
            previous_image_button.disable()
            next_image_button.disable()
            confidence_threshhold_input.disable()
            nms_threshhold_input.disable()
            random_image_button.disable()
            model_input_tabs.disable()
    if 'preview_card' in cards:
        if enabled:
            preview_card.enable()
            grid_gallery.enable()
            update_images_preview_button.enable()
            update_predictions_preview_button.enable()
        else:
            preview_card.disable()
            grid_gallery.disable()
            update_images_preview_button.disable()
            update_predictions_preview_button.disable()
    if 'run_model_card' in cards:
        if enabled:
            run_model_card.enable()
            keep_existed_annotations.enable()
            output_project_name_input.enable()
            select_output_destination.enable()
        else:
            run_model_card.disable()
            keep_existed_annotations.disable()
            output_project_name_input.disable()
            select_output_destination.disable()
    
toggle_cards(['inference_type_selection_card', 'model_settings_card', 'preview_card', 'run_model_card'], enabled=False)
set_model_type_button.disable()
set_input_button.disable()
run_model_button.disable()

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
