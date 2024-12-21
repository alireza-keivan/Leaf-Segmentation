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
```python
def filter_contours_by_shape(mask):
    
    # Finding contours 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the filtered contours
    leaf_mask = np.zeros_like(mask)

    # contours based on size and shape
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:  # Filter based on area
            # Approximate the contour to reduce noise
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Use the number of vertices to estimate the shape (leaf-like shapes are usually smooth)
            if len(approx) > 5:  # Ignore shapes that are too simple (like lines or small blobs)
                cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return leaf_mask
```
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
```python
def detect_blight(image, leaf_mask):
    
    # image to HSV 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV range for brown/orange blighted leaves
    lower_blighted = np.array([10, 40, 40])
    upper_blighted = np.array([25, 255, 255])

    # mask for blighted regions
    blight_mask = cv2.inRange(hsv, lower_blighted, upper_blighted)

    # Filter the blight mask by the leaf mask (to ensure blights are within the leaf region)
    blight_mask = cv2.bitwise_and(blight_mask, leaf_mask)

    # morphology operation for cleaning
    kernel = np.ones((5, 5), np.uint8)
    blight_mask = cv2.morphologyEx(blight_mask, cv2.MORPH_CLOSE, kernel)  # Close small holes

    return blight_mask
```

#### `draw_circles_around_blight(image, blight_mask)`
- **Input**: Original image and the binary mask of blighted regions.
- **Output**: Original image with red circles drawn around blighted regions.
- **Purpose**: Highlights blighted regions for visualization.
- **Steps**:
  1. Detects contours of the blighted regions.
  2. Computes the minimum enclosing circle for each contour.
  3. Draws red circles on the original image around the blighted areas.
```python
def draw_circles_around_blight(image, blight_mask):

    # Find contours of the blighted regions
    contours, _ = cv2.findContours(blight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around each blighted region
    for contour in contours:
        # Calculate the minimum enclosing circle for each blighted region
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Drawing circles on the original image
        center = (int(x), int(y))
        radius = int(radius)

        # Draw a thin circle 
        cv2.circle(image, center, radius, (0, 0, 255), thickness=2)  # Red circles

    return image
```


#### `remove_background(image, mask)`
- **Input**: Original image and the leaf mask.
- **Output**: The image with the background removed.
- **Purpose**: Removes the background, keeping only the leaf regions.
- **Steps**:
  1. Converts the binary mask to a 3-channel image.
  2. Applies the mask to the original image using bitwise operations.
 ```python
def remove_background(image, mask):

    # Convert the mask to a 3-channel image
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image to keep the leaves only
    result = cv2.bitwise_and(image, mask_3channel)

    return result
```

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

```python
def process_images(input_folder, output_folder):
    
    #output folder existence
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all images in the folder
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Could not open image: {file_path}")
                continue

            print(f"Processing image: {file_path}")

            # Segment the leaf from the image
            initial_mask = segment_leaf(image)

            # Filter the mask 
            leaf_mask = filter_contours_by_shape(initial_mask)

            # Detect blighted regions
            blight_mask = detect_blight(image, leaf_mask)

            # Draw circles around the blighted regions
            image_with_circles = draw_circles_around_blight(image, blight_mask)

            # Remove the background
            result = remove_background(image_with_circles, leaf_mask)

            # output path
            output_path = os.path.join(output_folder, os.path.relpath(file_path, input_folder))
            output_dir = os.path.dirname(output_path)

            # existence of the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Saving the result
            if not cv2.imwrite(output_path, result):
                print(f"Failed to save image: {output_path}")
            else:
                print(f"Processed and saved: {output_path}")
```

### Folder Structure

#### Input Folder
- Contains the raw images to be processed.
- Example: `PlantDoc-Dataset/train` and `PlantDoc-Dataset/test`.

#### Output Folder
- Stores the processed images with the same folder structure as the input.
- Example: `PlantDoc-Dataset/output`.
## Output Examples
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/raw%20images/Apple_RustSpots2.jpg)
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/segmented%20images/Apple_RustSpots2.jpg)
---
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/raw%20images/image_1500x1500%25253E.jpeg_1472603885.jpg)
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/segmented%20images/image_1500x1500%25253E.jpeg_1472603885.jpg)
---
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/raw%20images/may27pepleaes002.jpg)
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/segmented%20images/may27pepleaes002.jpg)
---
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/raw%20images/37356969601_4208f8bb7f_b.jpg)
![image](https://github.com/alireza-keivan/Leaf-Segmentation/blob/main/src/segmented%20images/37356969601_4208f8bb7f_b.jpg)
## For more Outputs check [src folder](https://github.com/alireza-keivan/Leaf-Segmentation/tree/main/src) 
## Dataset

- **Source**: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Description**: A dataset containing plant leaf images with various diseases, including blight.
- **Structure**:
  - `train`: Training images.
  - `test`: Testing images.
```python
# Example usage:
train_folder = "P:/1-uni/machine-learning/PlantDoc-Dataset/train"
test_folder = "P:/1-uni/machine-learning/PlantDoc-Dataset/test"
output_folder = "P:/1-uni/machine-learning/PlantDoc-Dataset/output"

# Process the train folder
process_images(train_folder, output_folder + "/train")

# Process the test folder
# process_images(test_folder, output_folder + "/test")
```
---


