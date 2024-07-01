import random
from typing import List

from src.utils import run

import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
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
    SelectDataset,
    DestinationProject,
    Flexbox,
    Grid,
    Empty,
    Table,
)

import src.sly_globals as g
from src.model import inference_json_anno_preprocessing

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
        if len(datasets_list) == 1:
            samples_count = dataset.images_count
        else:
            samples_count = dataset.images_count * (100 - len(datasets_list)) // 100
        if samples_count == 0:
            break

        IMAGES_INFO_LIST += random.sample(g.api.image.get_list(dataset.id), samples_count)
        if len(IMAGES_INFO_LIST) >= 1000:
            break
    return IMAGES_INFO_LIST


IMAGES_INFO_LIST = get_images_infos_for_preview()

######################
### Input project card
######################
dataset_selector = SelectDataset(
    project_id=g.project_id, multiselect=True, select_all_datasets=True
)


# def func_caller(value):
#    on_dataset_selected(value)


def on_dataset_selected(new_dataset_ids=None):
    global IMAGES_INFO_LIST, CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY

    if not new_dataset_ids:
        new_dataset_ids = dataset_selector.get_selected_ids()

    new_project_id = dataset_selector._project_selector.get_selected_id()
    if new_project_id != g.project_id:
        dataset_selector.disable()
        g.project_id = new_project_id
        g.project_info = g.api.project.get_info_by_id(g.project_id)
        g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
        g.workspace = g.api.workspace.get_info_by_id(g.project_info.workspace_id)
        dataset_selector.enable()
        g.DATASET_IDS = new_dataset_ids
        IMAGES_INFO_LIST = get_images_infos_for_preview()
        # update_images_preview()
        CURRENT_REF_IMAGE_INDEX = 0
        REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
        image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])
    else:
        if set(g.DATASET_IDS) != set(new_dataset_ids):
            g.DATASET_IDS = new_dataset_ids
            IMAGES_INFO_LIST = get_images_infos_for_preview()
            # update_images_preview()
            CURRENT_REF_IMAGE_INDEX = 0
            REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
            image_region_selector.image_update(IMAGES_INFO_LIST[CURRENT_REF_IMAGE_INDEX])

    sly.logger.info(
        f"Team: {g.team.id} \t Project: {g.project_info.id} \t Datasets: {g.DATASET_IDS}"
    )

    if len(new_dataset_ids) == 0:
        button_download_data.disable()
    else:
        button_download_data.enable()


dataset_selector.value_changed(on_dataset_selected)
button_download_data = Button("Select data")


@button_download_data.click
def download_data():
    on_dataset_selected()

    if data_card.is_disabled() is True:
        toggle_cards(["data_card"], enabled=True)
        toggle_cards(
            [
                "inference_type_selection_card",
                "model_settings_card",
                "preview_card",
                "run_model_card",
            ],
            enabled=False,
        )
        button_download_data.enable()
        button_download_data.text = "Select data"
        set_model_type_button.disable()
        set_model_type_button.text = "Select model"
        set_input_button.disable()
        set_input_button.text = "Set model input"
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        input_project_thmb.hide()
        run_model_button.disable()
        stepper.set_active_step(1)
    else:
        button_download_data.disable()
        toggle_cards(
            [
                "data_card",
                "inference_type_selection_card",
                "model_settings_card",
                "preview_card",
                "run_model_card",
            ],
            enabled=False,
        )

        build_table()

        input_project_thmb.set(info=g.project_info)
        input_project_thmb.show()
        button_download_data.text = "Change data"
        toggle_cards(["inference_type_selection_card"], enabled=True)
        if select_model_session.get_selected_id() is not None:
            set_model_type_button.enable()
        stepper.set_active_step(2)

        button_download_data.enable()
        progress_bar_download_data.hide()


progress_bar_download_data = Progress(hide_on_finish=False)
progress_bar_download_data.hide()
input_project_thmb = ProjectThumbnail()
input_project_thmb.hide()
data_card = Card(
    title="Input data selection",
    content=Container(
        [
            dataset_selector,
            progress_bar_download_data,
            input_project_thmb,
            button_download_data,
        ]
    ),
)


############################
### Inference type selection
############################
select_model_session = SelectAppSession(
    team_id=g.team.id, tags=["deployed_owl_vit_object_detection"]
)


@select_model_session.value_changed
def select_model_session_change(val):
    if val is None:
        set_model_type_button.disable()
    else:
        set_model_type_button.enable()


model_info = ModelInfo()
model_set_done = DoneLabel("Model successfully connected.")
model_set_done.hide()
set_model_type_button = Button(text="Select model")


@set_model_type_button.click
def set_model_type():
    global MODEL_DATA
    if set_model_type_button.text == "Disconnect model":
        MODEL_DATA["session_id"] = None
        MODEL_DATA["model_meta"] = None
        model_set_done.hide()
        model_info.hide()
        select_model_session.enable()
        set_model_type_button.enable()
        set_model_type_button.text = "Connect to model"
        toggle_cards(["model_settings_card", "preview_card", "run_model_card"], enabled=False)
        set_input_button.disable()
        set_input_button.text = "Set model input"
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        run_model_button.disable()
        stepper.set_active_step(2)
    else:
        model_session_id = select_model_session.get_selected_id()
        if model_session_id is not None:
            try:
                set_model_type_button.disable()
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
                model_set_done.text = "Model successfully connected."
                model_set_done.show()
                model_info.set_session_id(session_id=model_session_id)
                model_info.show()
                set_model_type_button.text = "Disconnect model"
                set_model_type_button._plain = True
                set_model_type_button.enable()
                toggle_cards(["model_settings_card"], enabled=True)
                set_input_button.enable()
                stepper.set_active_step(3)
            except Exception as e:
                sly.app.show_dialog(
                    "Error",
                    f"Cannot to connect to model. Make sure that model is deployed and try again.",
                    status="error",
                )
                sly.logger.warn(f"Cannot to connect to model. {e}")
                set_model_type_button.enable()
                set_model_type_button.text = "Connect to model"
                model_set_done.hide()
                select_model_session.enable()
                model_info.hide()
                toggle_cards(
                    ["model_settings_card", "preview_card", "run_model_card"],
                    enabled=False,
                )
                set_input_button.disable()
                update_images_preview_button.disable()
                update_predictions_preview_button.disable()
                run_model_button.disable()
                stepper.set_active_step(2)


inference_type_selection_card = Card(
    title="Connect to model",
    description="Select served model from list below",
    content=Container([select_model_session, model_info, model_set_done, set_model_type_button]),
)


#############################
### Model input configuration
#############################
text_prompt_textarea = Input(placeholder="blue car;dog;white rabbit;seagull")
text_prompt_textarea_field = Field(
    content=text_prompt_textarea,
    title="Describe what do you want to detect via NN",
    description="Names and descriptions of objects. For many objects use ; as separator.",
)
class_input = Input(placeholder="The class name for selected object")
class_input_field = Field(
    content=class_input,
    title="Class name",
    description="All detected objects will be added to project/dataset with this class name",
)
image_region_selector = ImageRegionSelector(
    widget_width="550px", widget_height="550px", points_disabled=True
)


@image_region_selector.bbox_changed
def bbox_updated(new_scaled_bbox):
    sly.logger.info(f"new_scaled_bbox: {new_scaled_bbox}")


previous_image_button = Button(
    "Previous image", icon="zmdi zmdi-skip-previous", button_size="small"
)
next_image_button = Button("Next image", icon="zmdi zmdi-skip-next", button_size="small")
random_image_button = Button("New random image", icon="zmdi zmdi-refresh", button_size="small")
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
    global IS_IMAGE_PROMPT
    if model_settings_card.is_disabled() is True:
        set_input_button.text = "Set model input"
        toggle_cards(["model_settings_card"], enabled=True)
        toggle_cards(["preview_card", "run_model_card"], enabled=False)
        update_images_preview_button.disable()
        update_predictions_preview_button.disable()
        run_model_button.disable()
        stepper.set_active_step(3)
    else:
        if IS_IMAGE_PROMPT and class_input.get_value().strip() == "":
            sly.app.show_dialog("Wrong value", "Class name can not be empty.", status="error")
            sly.logger.warning("Class name can not be empty.")
            return
        elif not IS_IMAGE_PROMPT and text_prompt_textarea.get_value().strip() == "":
            sly.app.show_dialog("Wrong value", "Text prompt can not be empty.", status="error")
            sly.logger.warning("Text prompt can not be empty.")
            return
        else:
            set_input_button.text = "Change model input"
            update_images_preview()
            toggle_cards(["model_settings_card"], enabled=False)
            toggle_cards(["preview_card", "run_model_card"], enabled=True)
            update_images_preview_button.enable()
            update_predictions_preview_button.enable()
            run_model_button.enable()
            stepper.set_active_step(5)


images_table = Table(fixed_cols=2, sort_column_id=1, per_page=15)
columns = [
    "DATASET NAME",
    "IMAGE ID",
    "FILE NAME",
    "IMAGE WIDTH",
    "IMAGE HEIGHT",
    "SELECT",
]


def build_table():
    global images_table, columns

    sly.logger.debug("Trying to build images table...")

    images_table.loading = True
    images_table.read_json(None)

    rows = []
    for dataset_id in g.DATASET_IDS:
        image_infos = g.api.image.get_list(dataset_id)
        dataset_info = g.api.dataset.get_info_by_id(dataset_id)

        for image_info in image_infos:
            rows.append(
                [
                    dataset_info.name,
                    image_info.id,
                    image_info.name,
                    image_info.width,
                    image_info.height,
                    sly.app.widgets.Table.create_button("SELECT"),
                ]
            )

    table_data = {"columns": columns, "data": rows}

    images_table.read_json(table_data)

    sly.logger.debug(f"Successfully built images table with {len(rows)} rows.")

    images_table.loading = False


@images_table.click
def handle_table_button(datapoint: sly.app.widgets.Table.ClickedDataPoint):
    if datapoint.button_name != "SELECT":
        return

    global IMAGES_INFO_LIST

    for image_info in IMAGES_INFO_LIST:
        if image_info.id == datapoint.row["IMAGE ID"]:
            image_region_selector.image_update(image_info)


model_input_tabs = RadioTabs(
    titles=["Reference image", "Text prompt"],
    contents=[
        Container(
            [
                Grid(
                    columns=2,
                    widgets=[
                        images_table,
                        Container(
                            [
                                Flexbox(
                                    [
                                        previous_image_button,
                                        next_image_button,
                                        random_image_button,
                                    ],
                                    gap=10,
                                    center_content=False,
                                ),
                                image_region_selector,
                            ]
                        ),
                    ],
                ),
                Grid([class_input_field, Empty()], columns=2, gap=5),
            ]
        ),
        Container([text_prompt_textarea_field]),
    ],
    descriptions=[
        "Pick object by bounding box editing",
        "Describe object, that you want to detect",
    ],
)


@model_input_tabs.value_changed
def model_input_changed(val):
    global IS_IMAGE_PROMPT
    IS_IMAGE_PROMPT = True if val == "Reference image" else False
    if val == "Reference image":
        confidence_threshhold_input.value = 0.8
    else:
        confidence_threshhold_input.value = 0.1


confidence_threshhold_input = InputNumber(value=0.8, min=00.1, max=1, step=0.01)
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
    content=Container(
        [
            model_input_tabs,
            Container(
                [field_confidence_threshhold, field_nms_threshhold],
                direction="horizontal",
            ),
            set_input_button,
        ]
    ),
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
grid_gallery.hide()

update_images_preview_button = Button("New random images", icon="zmdi zmdi-refresh")


@update_images_preview_button.click
def update_images_preview():
    update_predictions_preview_button.disable()
    global IMAGES_INFO_LIST

    grid_gallery.clean_up()
    NEW_PREVIEW_IMAGES_INFOS = []

    preview_images_number = preview_images_number_input.get_value()

    grid_gallery.columns_number = min(preview_images_number, g.COLUMNS_COUNT)
    grid_gallery._update_layout()

    for i in range(preview_images_number):
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

    grid_gallery.show()


update_predictions_preview_button = Button("Predictions preview", icon="zmdi zmdi-labels")


@update_predictions_preview_button.click
def update_predictions_preview():
    global IS_IMAGE_PROMPT
    update_images_preview_button.disable()
    confidence_threshold = confidence_threshhold_input.get_value()
    nms_threshold = nms_threshhold_input.get_value()

    # for TEXT PROMPT
    text_queries = text_prompt_textarea.get_value().split(";")
    # for IMAGE REFERENCE
    selected_bbox = image_region_selector.get_bbox()
    x0, y0, x1, y1 = *selected_bbox[0], *selected_bbox[1]

    annotations_list = []

    with preview_progress(
        message="Generating predictions...", total=len(PREVIEW_IMAGES_INFOS)
    ) as pbar:
        for i, image_info in enumerate(PREVIEW_IMAGES_INFOS):
            if IS_IMAGE_PROMPT:
                inference_settings = dict(
                    mode="reference_image",
                    reference_bbox=[y0, x0, y1, x1],
                    reference_image_id=image_region_selector._image_id,
                    reference_class_name=class_input.get_value(),
                    confidence_threshold=[
                        {"text_prompt": confidence_threshold},
                        {"reference_image": confidence_threshold},
                    ],
                    nms_threshold=nms_threshold,
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
                    mode="text_prompt",
                    text_queries=text_queries,
                    confidence_threshold=[
                        {"text_prompt": confidence_threshold},
                        {"reference_image": confidence_threshold},
                    ],
                    nms_threshold=nms_threshold,
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

            pbar.update(1)

    grid_gallery.clean_up()
    for i, (image_info, annotation) in enumerate(zip(PREVIEW_IMAGES_INFOS, annotations_list)):
        grid_gallery.append(
            image_url=image_info.preview_url,
            annotation=annotation,
            title=image_info.name,
            column_index=int(i % g.COLUMNS_COUNT),
        )
    update_images_preview_button.enable()


preview_images_number_input = InputNumber(value=12, min=1, max=60, step=1)
preview_images_number_field = Field(
    title="Number of images in preview",
    description="Select how many images should be in preview gallery",
    content=preview_images_number_input,
)


@preview_images_number_input.value_changed
def preview_images_number_changed(preview_images_number):
    if not grid_gallery._data:
        sly.logger.debug("Preview gallery is empty, nothing to update.")
        return

    update_images_preview()


preview_progress = Progress()

preview_buttons_flexbox = Flexbox(
    widgets=[
        update_images_preview_button,
        update_predictions_preview_button,
    ],
)


preview_card = Card(
    title="Preview results",
    description="Model prediction result preview",
    content=Container(
        [
            preview_images_number_field,
            preview_buttons_flexbox,
            preview_progress,
            grid_gallery,
        ]
    ),
)


#######################
### Applying model card
#######################


destination_project = DestinationProject(g.workspace.id, project_type=sly.ProjectType.IMAGES)
run_model_button = Button("Run model")


@run_model_button.click
def run_model():
    toggle_cards(
        [
            "data_card",
            "inference_type_selection_card",
            "model_settings_card",
            "preview_card",
            "run_model_card",
        ],
        enabled=False,
    )
    button_download_data.disable()
    set_input_button.disable()
    next_image_button.disable()
    random_image_button.disable()
    previous_image_button.disable()
    set_model_type_button.disable()
    update_images_preview_button.disable()
    update_predictions_preview_button.disable()
    output_project_thmb.hide()
    global IS_IMAGE_PROMPT, MODEL_DATA
    confidence_threshold = confidence_threshhold_input.get_value()
    nms_threshold = nms_threshhold_input.get_value()

    def get_inference_settings():
        if IS_IMAGE_PROMPT:
            selected_bbox = image_region_selector.get_bbox()
            x0, y0, x1, y1 = *selected_bbox[0], *selected_bbox[1]
            inference_settings = dict(
                mode="reference_image",
                reference_bbox=[y0, x0, y1, x1],
                reference_image_id=image_region_selector._image_id,
                reference_class_name=class_input.get_value(),
                confidence_threshold=[
                    {"text_prompt": confidence_threshold},
                    {"reference_image": confidence_threshold},
                ],
                nms_threshold=nms_threshold,
            )
        else:
            text_queries = text_prompt_textarea.get_value().split(";")
            inference_settings = dict(
                mode="text_prompt",
                text_queries=text_queries,
                confidence_threshold=[
                    {"text_prompt": confidence_threshold},
                    {"reference_image": confidence_threshold},
                ],
                nms_threshold=nms_threshold,
            )
        return inference_settings

    try:
        output_project_info = run(destination_project, get_inference_settings(), MODEL_DATA)

        output_project_thmb.set(output_project_info)
        output_project_thmb.show()
        sly.logger.info("Project was successfully labeled")
    except Exception as e:
        sly.logger.error(f"Something went wrong. Error: {e}")
    finally:
        toggle_cards(["run_model_card"], enabled=True)
        button_download_data.enable()
        set_input_button.enable()
        set_model_type_button.enable()
        run_model_button.enable()


output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
run_model_card = Card(
    title="Apply model",
    content=Container([destination_project, run_model_button, output_project_thmb]),
)


def toggle_cards(cards: List[str], enabled: bool = False):
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY

    def set_card_state(card, state, elements=[]):
        if state:
            card.enable()
            for element in elements:
                element.enable()
        else:
            card.disable()
            for element in elements:
                element.disable()

    card_mappings = {
        "data_card": (data_card, [dataset_selector]),
        "inference_type_selection_card": (inference_type_selection_card, [select_model_session]),
        "model_settings_card": (
            model_settings_card,
            [
                text_prompt_textarea,
                class_input,
                image_region_selector,
                confidence_threshhold_input,
                nms_threshhold_input,
                random_image_button,
                model_input_tabs,
                previous_image_button if CURRENT_REF_IMAGE_INDEX != REF_IMAGE_HISTORY[0] else None,
                next_image_button if CURRENT_REF_IMAGE_INDEX < len(IMAGES_INFO_LIST) - 1 else None,
            ],
        ),
        "preview_card": (
            preview_card,
            [grid_gallery, update_images_preview_button, update_predictions_preview_button],
        ),
        "run_model_card": (run_model_card, [destination_project]),
    }

    for card in cards:
        if card in card_mappings:
            card_element, elements = card_mappings[card]
            set_card_state(card_element, enabled, [e for e in elements if e is not None])


toggle_cards(
    [
        "inference_type_selection_card",
        "model_settings_card",
        "preview_card",
        "run_model_card",
    ],
    enabled=False,
)
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
