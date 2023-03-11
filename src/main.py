import os
import sys
from typing import List
import random

import jax
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
    TextArea,
    GridGallery,
    InputNumber,
    Field,
    Progress,
)
import src.sly_globals as g

sys.path.append(os.path.join(g.app_root_directory, "scenic"))
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32
from src.model import prepare_image, prepare_text, get_predictions, draw_predictions


PREVIEW_IMAGES_INFOS = []
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

ref_image_info = image_info_list[0]


def get_image_path(image_name: str) -> str:
    for dataset_name, dataset in g.project_fs.datasets:
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
                message=f"Processing images...", total=g.project.items_count
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


###########################################
### Model settings and results preview card
###########################################
text_prompt_textarea = TextArea(
    placeholder="Description of object, that you want to detect via NN model", rows=10
)
image_region_selector = ImageRegionSelector(
    image_info=ref_image_info, widget_width="500px", widget_height="500px"
)


@image_region_selector.bbox_changed
def bbox_updated(new_scaled_bbox):
    sly.logger.info(f"new_scaled_bbox: {new_scaled_bbox}")


previous_image_button = Button("Previous image", icon="zmdi zmdi-skip-previous")
next_image_button = Button("Next image", icon="zmdi zmdi-skip-next")
random_image_button = Button("New random image", icon="zmdi zmdi-refresh")
set_input_button = Button("Set model input")


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
                image_region_selector,
                previous_image_button,
                next_image_button,
                random_image_button,
            ]
        ),
        Container([text_prompt_textarea]),
    ],
    descriptions=[
        "Pick object by bounding box editing",
        "Describe object, that you want to detect",
    ],
)
model_settings_card = Card(
    title="Model settings",
    description="Configure input for model as text-prompt or as reference image",
    content=Container([model_input_tabs, set_input_button]),
)

grid_gallery = GridGallery(
    columns_number=g.PREVIEW_IMAGES_COUNT,
    annotations_opacity=0.5,
    show_opacity_slider=True,
    enable_zoom=False,
    sync_views=False,
    fill_rectangle=True,
)
for i in range(g.PREVIEW_IMAGES_COUNT):
    img_info = random.choice(image_info_list)
    PREVIEW_IMAGES_INFOS.append(img_info)
    grid_gallery.append(
        title=img_info.name,
        image_url=img_info.preview_url,
        column_index=int(i % 3),
    )
confidence_threshhold_input = InputNumber(value=0.5, min=0.1, max=1, step=0.1)
nms_threshhold_input = InputNumber(value=0.5, min=0.1, max=1, step=0.1)
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
update_preview_button = Button("Update preview")


@update_preview_button.click
def update_preview():
    confidence_threshhold = confidence_threshhold_input.get_value()
    nms_threshhold = nms_threshhold_input.get_value()

    config = clip_b32.get_config(init_mode="canonical_checkpoint")
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )
    variables = module.load_variables(config.init_from.checkpoint_path)
    # model = inference.Model(config, module, variables)
    # model.warm_up()

    query_image = sly.image.read(get_image_path(ref_image_info))

    x0, y0, x1, y1 = np.array(image_region_selector.scaled_bbox).reshape(-1)
    box = [y1, x0, y0, x1]
    query_embedding, best_box_ind = module.embed_image_query(query_image, box)

    for i, image_info in enumerate(PREVIEW_IMAGES_INFOS):
        target_image = sly.image.read(get_image_path(image_info))
        input_image = prepare_image(target_image)

        if model_input_tabs.get_active_tab() == "Reference image":
            selected_bbox = image_region_selector.get_relative_coordinates()
        else:
            text_prompt = text_prompt_textarea.get_value().split(";")
            tokenized_queries = prepare_text(text_prompt)
            # Note: The model expects a batch dimension.
            predictions = module.apply(
                variables,
                input_image[None, ...],
                tokenized_queries[None, ...],
                train=False,
            )

            # Remove batch dimension and convert to numpy:
            predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)

        draw_predictions(predictions, confidence_threshhold, nms_threshhold)
        sly.logger.info(
            f"{i+1} image processed. {len(PREVIEW_IMAGES_INFOS) - (i+1)} images left."
        )


preview_card = Card(
    title="Preview results",
    description="Model prediction result preview",
    content=Container(
        [
            field_confidence_threshhold,
            field_nms_threshhold,
            grid_gallery,
            update_preview_button,
        ]
    ),
)

run_model_button = Button("Run model")
model_progress = Progress(message="Applying model..", hide_on_finish=False)
run_model_card = Card(
    title="Model apply progress",
    content=Container([run_model_button, model_progress]),
)

stepper = Stepper(
    widgets=[data_card, model_settings_card, preview_card, run_model_card]
)
app = sly.Application(layout=stepper)
