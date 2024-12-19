# Leaf Segmentation
# Leaf Blight Detection and Segmentation

## Overview

This project implements an image processing pipeline to detect blight symptoms on leaves, segment healthy leaf regions, and highlight blighted areas. The goal is to automate plant disease detection using computer vision techniques, focusing on leaf segmentation and detection of blight using color thresholds in the HSV color space.

The pipeline processes images from the PlantDoc dataset, identifies leaf regions, detects blight, and saves the processed results in an output directory.

---

## Features

- **Leaf Segmentation**: Identifies and isolates leaf regions from the background.
- **Blight Detection**: Detects blight-affected regions on the leaves using HSV color space thresholds.
- **Blight Visualization**: Draws red circles around detected blighted areas for clarity.
- **Background Removal**: Removes the background while preserving only leaf regions.
- **Batch Processing**: Processes all images in a directory structure and saves the results in a structured output folder.

---

## File and Code Structure

### Core Functions

#### `filter_contours_by_shape(mask)`
- **Input**: A binary mask of the segmented leaf.
- **Output**: A refined mask containing only the contours that resemble leaves.
- **Purpose**: Filters out small or irregular shapes by checking contour area and approximating the shape.
- **Steps**:
  1. Finds contours in the mask.
  2. Filters contours with an area threshold (>1000 pixels).
  3. Approximates the contour shape and retains smooth shapes with more than 5 vertices.
  4. Draws the filtered contours onto a new mask.

#### `detect_blight(image, leaf_mask)`
- **Input**: Original image and the refined leaf mask.
- **Output**: A binary mask highlighting blighted regions.
- **Purpose**: Identifies blighted regions in leaves using HSV color thresholds.
- **Steps**:
  1. Converts the image to the HSV color space.
  2. Defines a color range for brown/orange blighted regions.
  3. Creates a mask for blighted regions using the HSV range.
  4. Refines the blight mask by combining it with the leaf mask.
  5. Cleans the mask using morphological operations.

#### `draw_circles_around_blight(image, blight_mask)`
- **Input**: Original image and the binary mask of blighted regions.
- **Output**: Original image with red circles drawn around blighted regions.
- **Purpose**: Highlights blighted regions for visualization.
- **Steps**:
  1. Detects contours of the blighted regions.
  2. Computes the minimum enclosing circle for each contour.
  3. Draws red circles on the original image around the blighted areas.

#### `remove_background(image, mask)`
- **Input**: Original image and the leaf mask.
- **Output**: The image with the background removed.
- **Purpose**: Removes the background, keeping only the leaf regions.
- **Steps**:
  1. Converts the binary mask to a 3-channel image.
  2. Applies the mask to the original image using bitwise operations.

#### `process_images(input_folder, output_folder)`
- **Input**: Input folder containing images and the output folder to save processed results.
- **Output**: Processed images saved in the output folder.
- **Purpose**: Processes all images in the input folder and applies the pipeline.
- **Steps**:
  1. Recursively iterates through the input folder.
  2. For each image:
     - Segments the leaf using `segment_leaf()` (to be implemented).
     - Filters the leaf mask using `filter_contours_by_shape()`.
     - Detects blight using `detect_blight()`.
     - Draws circles around blighted regions using `draw_circles_around_blight()`.
     - Removes the background using `remove_background()`.
  3. Saves the processed image to the output folder, maintaining the input folder’s structure.

---

### Folder Structure

#### Input Folder
- Contains the raw images to be processed.
- Example: `PlantDoc-Dataset/train` and `PlantDoc-Dataset/test`.

#### Output Folder
- Stores the processed images with the same folder structure as the input.
- Example: `PlantDoc-Dataset/output`.

---

## Dataset

- **Source**: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Description**: A dataset containing plant leaf images with various diseases, including blight.
- **Structure**:
  - `train`: Training images.
  - `test`: Testing images.

---

## Usage

### Requirements

- Install the required Python libraries:
  ```bash
  pip 
