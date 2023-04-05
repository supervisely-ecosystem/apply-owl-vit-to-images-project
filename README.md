<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/230065782-f033aa7d-6acc-405b-89e3-81330bb66bbb.png" />

# Apply Owl-ViT to Images Project
Integration of the Owl-ViT model for class-agnostic object detection

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#Demo">Demo</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/apply-owl-vit-to-images-project)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/apply-owl-vit-to-images-project)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/apply-owl-vit-to-images-project.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/apply-owl-vit-to-images-project.png)](https://supervise.ly)

</div>

## Overview
Application allows you to label projects images using Owl-ViT detection model.

Application key points:

- Select project to label
- Choose the model configuration for local run or run via [serve owl-vit app](https://github.com/supervisely-ecosystem/serve-owl-vit) 
- Set up model input data as text-prompt or reference-image
- Preview detection results
- Apply model to project images and save new annotations to new project or add to existed  

## How to Run

1. Start the application from Ecosystem or context menu of an Images Project

2. Choose your input project / dataset

<img src="https://user-images.githubusercontent.com/115161827/230068919-1c67170a-855f-4372-b78c-823c0a4da0fd.png" />

3. Select local inference / served model and click `Select model` button

<img src="https://user-images.githubusercontent.com/115161827/230068913-f42c2242-3a8d-4aef-86f2-53eeb61fedb8.png" />

4. **Reference Image**: <br> pick reference image from your project by clicking navigating buttons and place the reference object in the bounding box <br>
   **Text Prompt**: <br> type the description of the object you want to detect in the corresponding field
  
Adjust the confidence and NMS thresholds and click `Set model input` button

<img src="https://user-images.githubusercontent.com/115161827/230068907-03be9de7-d75a-4649-8c7e-9bc4fada9731.png" />

5. View predictions preview by clicking according button

<img src="https://user-images.githubusercontent.com/115161827/230068878-18d2ef04-f7fa-47b8-8c58-b42c2298c687.png" />

6. Select the way you want to save the project and click `Run model`

<img src="https://user-images.githubusercontent.com/115161827/230068872-07e9bdff-63f6-4f6c-8095-d00119148831.png" />

## Results screenshoot
<details>
<summary>Reference-image</summary>
<img src="docs/images/screenshoot.png" />
</details>

<details>
<summary>Text-prompt</summary>
<img src="docs/images/screenshoot.png" />
</details>
