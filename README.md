<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/230065782-f033aa7d-6acc-405b-89e3-81330bb66bbb.png" />

# Apply Owl-ViT to Images Project
Integration of the Owl-ViT model for class-agnostic object detection

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/apply-owl-vit-to-images-project)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/apply-owl-vit-to-images-project)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/apply-owl-vit-to-images-project.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/apply-owl-vit-to-images-project.png)](https://supervisely.com)

</div>

## Overview
Application allows you to label projects images using Owl-ViT detection model.

Application key points:

- Select project or dataset to label
- Serve this model by [Serve OWL-ViT](https://ecosystem.supervisely.com/apps/apply-object-segmentor-to-images-project) app and choose model session in selector
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/serve-owl-vit" src="https://user-images.githubusercontent.com/115161827/230909082-8d0cb4f1-ce7c-490a-9187-f87d061c1f7b.png" height="70px" margin-bottom="20px"/>

- Set up model input data as text-prompt or reference-image
- Preview detection results
- Apply model to project images and save new annotations to new project or add to existed

🔥🔥🔥 Check out our [youtube tutorial](https://youtu.be/PnhAsG-GFHo) and the [complete guide in our blog](https://supervisely.com/blog/owl-vit/):   

<a href="https://youtu.be/PnhAsG-GFHo" target="_blank"><img src="https://github.com/supervisely-ecosystem/serve-owl-vit/assets/12828725/f1a3589a-2c4d-4467-a2dc-077f56422a98"/></a>

## How to Run

1. Start the application from Ecosystem or context menu of an Images Project

2. Choose your input project / dataset

<img src="https://user-images.githubusercontent.com/115161827/230068919-1c67170a-855f-4372-b78c-823c0a4da0fd.png" />

3. Select your served model and click `Select model` button

<img src="https://user-images.githubusercontent.com/115161827/230893590-cb0077e0-7a4f-4ce4-9bc1-a70278e07b49.png" />

4. **Reference Image**: <br> &emsp; Pick reference image from your project by clicking navigating buttons and place the reference object in the bounding box </br> <br>
   **Text Prompt**: <br> &emsp; Type the description of the object you want to detect in the corresponding field
  
Adjust the confidence and NMS thresholds and click `Set model input` button

<img src="https://user-images.githubusercontent.com/115161827/230068907-03be9de7-d75a-4649-8c7e-9bc4fada9731.png" />

5. View predictions preview by clicking according button

<img src="https://user-images.githubusercontent.com/115161827/230946583-df804396-4e03-4fc1-8d33-13494d097dab.png" />

6. Select the way you want to save the project and click `Run model`

<img src="https://user-images.githubusercontent.com/115161827/230893629-f4ce9a55-8df9-47a5-a698-5c0c2055ea26.png" />
