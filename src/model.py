import os, sys
sys.path.insert(0,os.path.join(g.app_root_directory, 'scenic'))

import os

import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform

# config = clip_b32.get_config(init_mode='canonical_checkpoint')

# module = models.TextZeroShotDetectionModule(
#     body_configs=config.model.body,
#     normalize=config.model.normalize,
#     box_bias=config.model.box_bias)

# variables = module.load_variables(config.init_from.checkpoint_path)

def prepare_image(config):
    # Load example image:
    filename = os.path.join(skimage.data_dir, 'astronaut.png')
    image_uint8 = skimage_io.imread(filename)
    image = image_uint8.astype(np.float32) / 255.0

    # Pad to square with gray pixels on bottom and right:
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    # Resize to model input size:
    input_image = skimage.transform.resize(
        image_padded,
        (config.dataset_configs.input_size, config.dataset_configs.input_size),
        anti_aliasing=True)
    
def prepare_text(config, module):
    text_queries = ['human face', 'rocket', 'nasa badge', 'star-spangled banner']
    tokenized_queries = np.array([
        module.tokenize(q, config.dataset_configs.max_query_length)
        for q in text_queries
    ])

    # Pad tokenized queries to avoid recompilation if number of queries changes:
    tokenized_queries = np.pad(
        tokenized_queries,
        pad_width=((0, 100 - len(text_queries)), (0, 0)),
        constant_values=0)

def get_predictions(module, variables, input_image=None, tokenized_queries=None):
    # Note: The model expects a batch dimension.
    predictions = module.apply(
        variables,
        input_image[None, ...],
        tokenized_queries[None, ...],
        train=False)

    # Remove batch dimension and convert to numpy:
    predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions )
    return predictions

def draw_predictions(predictions, score_threshold: float = 0.1):
    logits = predictions['pred_logits'][..., :len(text_queries)]  # Remove padding.
    scores = sigmoid(np.max(logits, axis=-1))
    labels = np.argmax(predictions['pred_logits'], axis=-1)
    boxes = predictions['pred_boxes']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue
        cx, cy, w, h = box
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f'{text_queries[label]}: {score:1.2f}',
            ha='left',
            va='top',
            color='red',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'red',
                'boxstyle': 'square,pad=.3'
            })