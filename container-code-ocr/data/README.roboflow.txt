
Container Character Codes - v3 Augmented
==============================

This dataset was exported via roboflow.com on January 13, 2023 at 5:50 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2545 images.
Container-Codes are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 10 versions of each source image:
* Randomly crop between 0 and 40 percent of the image
* Random rotation of between -1 and +1 degrees
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -5 and +5 percent


