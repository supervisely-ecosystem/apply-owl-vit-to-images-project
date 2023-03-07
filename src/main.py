import supervisely as sly
from supervisely.app.widgets import (
    Button, Card, Progress, Text, Empty,
    Container, Checkbox, Stepper, ClassesTable, 
    ImageRegionSelector, ProjectThumbnail,
    RadioTabs, TextArea, GridGallery, InputNumber, Field, Progress
)

import src.sly_globals as g
import sys 
import os
sys.path.append(os.path.join(g.app_root_directory, "scenic"))

from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b16 as config_module
from scenic.projects.owl_vit.notebooks import inference
# from scenic.projects.owl_vit.notebooks import interactive
# from scenic.projects.owl_vit.notebooks import plotting
from scenic.model_lib.base_models import box_utils
import numpy as np

######################
### Input project card
######################
project_preview = ProjectThumbnail(g.project)
progress_bar_download_data = Progress(hide_on_finish=False)
progress_bar_download_data.hide()
text_download_data = Text('Data has been successfully downloaded', status='success')
text_download_data.hide()
button_download_data = Button('Download')
@button_download_data.click
def download_data():
    try:
        if sly.fs.dir_exists(g.project_dir):
            sly.logger.info('Data already downloaded.')
        else:
            button_download_data.hide()
            progress_bar_download_data.show()
            sly.fs.mkdir(g.project_dir)
            with progress_bar_download_data(message=f"Processing images...", total=g.project.items_count) as pbar:
                sly.Project.download(
                    api=g.api, 
                    project_id=g.project_id, 
                    dest_dir=g.project_dir,
                    batch_size=100,
                    progress_cb=pbar.update,
                    only_image_tags=False, 
                    save_image_info=True)
            sly.logger.info('Data successfully downloaded.')
        g.project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        progress_bar_download_data.hide()
        button_download_data.hide()
        text_download_data.show()
        stepper.set_active_step(2)
    except Exception as e:
        sly.logger.info('Something went wrong.')
        progress_bar_download_data.hide()
        button_download_data.show()
        text_download_data.set('Data download failed', status='error')
        text_download_data.show()
        stepper.set_active_step(1)
data_card = Card(
    title="Input data", 
    content=Container([project_preview, progress_bar_download_data, text_download_data, button_download_data])
)


###########################################
### Model settings and results preview card
###########################################
text_prompt_textarea = TextArea(placeholder='Description of object, that you want to detect via NN model', rows=10)
datasets_list = g.api.dataset.get_list(g.project_id)
image_info_list = g.api.image.get_list(datasets_list[0].id)
ref_image = image_info_list[0]
image_region_selector = ImageRegionSelector(image_info=ref_image, widget_width='500px', widget_height='500px')
@image_region_selector.bbox_changed
def bbox_updated(new_scaled_bbox):
    sly.logger.info(f"new_scaled_bbox: {new_scaled_bbox}")

previous_image_button = Button('Previous image', icon='zmdi zmdi-skip-previous')
next_image_button = Button('Next image', icon='zmdi zmdi-skip-next')
random_image_button = Button('New random image', icon='zmdi zmdi-refresh')
set_input_button = Button('Set model input')
model_input_tabs = RadioTabs(
    titles=['Reference image', 'Text prompt'], 
    contents=[Container([image_region_selector, previous_image_button, next_image_button, random_image_button]),
              Container([text_prompt_textarea])], 
    descriptions=["Pick object by bounding box editing", "Describe object, that you want to detect"]
)
model_settings_card = Card(
    title="Model settings",
    description="Configure input for model as text-prompt or as reference image",
    content=Container([model_input_tabs, set_input_button]),
)

run_model_button = Button('Run model')
grid_gallery = GridGallery(
    columns_number=3,
    annotations_opacity=0.5,
    show_opacity_slider=True,
    enable_zoom=False,
    sync_views=False,
    fill_rectangle=True,
)

def get_random_image(image_infos_list):
    from random import choice
    return choice(image_infos_list)

field_confidence_threshhold = Field(title='Confidence threshold', description='Threshold for the minimum confidence that a detection must have to be displayed (higher values mean fewer boxes will be shown):', content=InputNumber(value=0.5, min=0.1, max=1, step=0.1))
field_nms_threshhold = Field(title='NMS threshold', description='Threshold for non-maximum suppression of overlapping boxes (higher values mean more boxes will be shown)', content=InputNumber(value=0.5, min=0.1, max=1, step=0.1))
for i in range(6):
    img_info = get_random_image(image_info_list)
    grid_gallery.append(
        title=img_info.name, 
        image_url=img_info.full_storage_url, 
        # annotation=img_info.ann, 
        column_index=int(i % 3)
    )
update_preview_button = Button('Update preview')
@update_preview_button.click
def update_preview():
    config = config_module.get_config(init_mode='canonical_checkpoint')
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias)
    variables = module.load_variables(config.init_from.checkpoint_path)
    model = inference.Model(config, module, variables)
    model.warm_up()

    PREVIEW_IMAGES_COUNT = 6
    IMAGE_COND_MIN_CONF = 0.5
    IMAGE_COND_NMS_IOU_THRESHOLD = 0.5

    images_infos, _ = g.project_fs.get_train_val_splits_by_count(g.project_dir, train_count=PREVIEW_IMAGES_COUNT, val_count=0)
    query_image = sly.image.read(ref_image.img_path)
    
    x0,y0,x1,y1 = np.array(image_region_selector.scaled_bbox).reshape(-1)
    box = [y1,x0,y0,x1]
    query_embedding, best_box_ind = model.embed_image_query(query_image, box)

    for image_info in images_infos:
        target_image = sly.image.read(image_info.img_path)
        # TODO(mjlm): Implement multi-query image-conditioned detection.
        num_queries = 1
        top_query_ind, scores = model.get_scores(target_image, query_embedding[None, ...], num_queries=1)
        
        # Apply non-maximum suppression:
        if IMAGE_COND_NMS_IOU_THRESHOLD < 1.0:
            _, _, target_image_boxes = model.embed_image(target_image)
            target_boxes_yxyx = box_utils.box_cxcywh_to_yxyx(target_image_boxes, np)
            for i in np.argsort(-scores):
                if not scores[i]:
                    # This box is already suppressed, continue:
                    continue
                ious = box_utils.box_iou(
                    target_boxes_yxyx[None, [i], :],
                    target_boxes_yxyx[None, :, :],
                    np_backbone=np)[0][0, 0]
                ious[i] = -1.0  # Mask self-IoU.
                scores[ious > IMAGE_COND_NMS_IOU_THRESHOLD] = 0.0

        for bbox, confidence in zip(target_boxes_yxyx, scores):
            if confidence > IMAGE_COND_MIN_CONF:
                grid_gallery._data[i]
                
    if model_input_tabs.get_active_tab() == 'Reference image':
        selected_bbox = image_region_selector.get_relative_coordinates()
    else:
        text_prompt = text_prompt_textarea.get_value()
    

    
    
    pass
preview_card = Card(
    title="Preview results",
    description="Model prediction result preview",
    content=Container([field_confidence_threshhold, field_nms_threshhold, grid_gallery, update_preview_button]),
)

model_progress = Progress(message='Applying model..', hide_on_finish=False)
run_model_card = Card(
    title="Model apply progress",
    content=Container([run_model_button, model_progress]),
)

stepper = Stepper(widgets=[data_card, model_settings_card, preview_card, run_model_card])
app = sly.Application(layout=stepper)