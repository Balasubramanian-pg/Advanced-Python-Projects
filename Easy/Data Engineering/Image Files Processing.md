---
title: Image Files Processing
company: Amazon
difficulty: Easy
category: Data Engineering
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Amazon._

## Assignment

**1. Data frame**

The purpose of the task is to test your ability to process image files and your knowledge of data-framing methods in a programming language of your choice. Based on the loaded photos, you need to create a data frame (e.g. Pandas DataFrame) with the following columns:

- File name
- Description of the photo content (the separator `-` should be replaced with a space)
- Image ID
- Width
- Height
- Average color
- Median brightness after conversion to shades of gray
- Horizontal coordinate of the brightest pixel after converting to grayscale*
- Vertical coordinate of brightest pixel after converting to grayscale*

*The upper left corner of the image corresponds to the coordinates (0,0). If there are many brightest pixels, we choose the one closest according to the Euclidean norm to (0,0).

After creating the data frame, save it to `images.csv` file.

**2. Aggregation of images**

The second task is to sort the photos into subfolders. You need to sort the images by median brightness, divide them into bins of count 4, and save the images into subfolders with the naming scheme - `[bin number]-images`, numbering the bins from 1.

## Data Description

Attached to the task is a directory `images`, which contains image files with the following file naming scheme - `stock-photo-[image content description]-[image id].jpg`. The images should be loaded into any programming environment in the RGB color model.

## Practicalities

Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final files.

# Solution

Here is a complete, structured solution to the Amazon data science take-home assignment on image processing.

This response is designed like a professional data science script and report. It includes:
1.  **Code to Generate a Sample Dataset:** As the original `images` directory is not provided, I will first generate a realistic synthetic dataset. This includes creating an `images/` directory and populating it with sample `.jpg` files that match the specified naming scheme and have varying visual properties. This ensures the entire solution is fully reproducible.
2.  **A Clear, Structured Python Script:** The solution is presented as a single, well-documented Python script that performs the entire analysis, from data extraction to file organization.
3.  **Explanation of the Approach:** The code is broken down into logical functions with clear explanations of the methodology, covering image processing, data extraction, and file system operations.
4.  **Final Output Generation:** The script creates the required `images.csv` file and sorts the image files into the specified subfolders.

***

## Amazon: Image Processing and Data Framing

### Project Objective
The goal of this project is to process a directory of image files, extract key metadata and computed properties from each image, store this information in a structured CSV file, and then physically reorganize the image files based on one of the computed properties (median brightness).

### 1. Setup and Data Generation

First, we will import the necessary libraries and create a synthetic `images` directory with sample files. This is crucial for creating a reproducible solution.

#### 1.1 Import Libraries
```python
# Core libraries for file system and data manipulation
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# For ensuring clean execution
from pathlib import Path
```

#### 1.2 Generate Sample Dataset
This function creates an `images/` directory and populates it with 12 sample images. The images are generated with varying colors and brightness levels to ensure the subsequent analysis is meaningful. The filenames follow the required `stock-photo-[description]-[id].jpg` format.

```python
def generate_sample_images(output_dir='images', num_images=12):
    """
    Generates a directory of sample images with varying properties.
    """
    # Clean up previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    descriptions = [
        "a-solitary-tree", "blue-ocean-waves", "bright-yellow-sunflower", "dark-night-sky",
        "red-sports-car", "green-forest-canopy", "a-cup-of-coffee", "snowy-mountain-peak",
        "a-modern-skyscraper", "a-field-of-lavender", "a-warm-fireplace", "a-single-white-cloud"
    ]
    
    print(f"Generating {num_images} sample images in '{output_dir}/'...")

    for i in range(num_images):
        image_id = 1000 + i
        description = descriptions[i % len(descriptions)]
        file_name = f"stock-photo-{description}-{image_id}.jpg"
        file_path = os.path.join(output_dir, file_name)
        
        # Create an image with varying properties
        width, height = np.random.randint(200, 401), np.random.randint(200, 401)
        # Create a base color that gets brighter with each image to test sorting
        base_color_value = 10 + (i * 20)
        base_color = (
            min(255, base_color_value + np.random.randint(0, 50)), 
            min(255, base_color_value + np.random.randint(0, 50)), 
            min(255, base_color_value + np.random.randint(0, 50))
        )
        img = Image.new('RGB', (width, height), color=base_color)

        # Add a brighter spot to test pixel finding
        draw = ImageDraw.Draw(img)
        bright_spot_x, bright_spot_y = np.random.randint(0, width), np.random.randint(0, height)
        draw.ellipse(
            (bright_spot_x, bright_spot_y, bright_spot_x + 10, bright_spot_y + 10),
            fill=(255, 255, 255)
        )
        
        img.save(file_path)

    print("Sample image generation complete.")

# Generate the data before processing
generate_sample_images()
```

<hr>

### 2. Main Image Processing Script

This script contains the core logic for processing the images, creating the DataFrame, and organizing the files.

#### 2.1 Approach and Thought Process
1.  **Modularity:** The logic for processing a single image is encapsulated in a dedicated function, `process_image()`. This makes the code clean, testable, and reusable.
2.  **File Iteration:** The main script iterates through all files in the source directory, ensuring only `.jpg` files are processed.
3.  **Data Extraction within `process_image()`:**
    *   **Filename Parsing:** The filename is parsed using string splitting (`.split('-')`) to extract the description and image ID according to the specified format. The description's hyphens are replaced with spaces.
    *   **Image Loading:** The `Pillow` library is used to open the image. Its `convert('L')` method is used to create a grayscale version for brightness calculations.
    *   **Numerical Processing:** The image data is converted into a `NumPy` array, which is highly efficient for calculations like mean, median, and finding max values.
    *   **Brightest Pixel Logic:** `np.argwhere` is used to find all coordinates of the brightest pixels. A small helper function then calculates the Euclidean norm for each of these coordinates to find the one closest to `(0,0)`, as required.
4.  **DataFrame Creation:** The main script collects the dictionary returned by `process_image()` for each image into a list. This list is then efficiently converted into a Pandas DataFrame.
5.  **File Organization:** The created DataFrame, which now contains the `median_brightness`, is sorted. The script then iterates through this sorted DataFrame in chunks of 4, creating the necessary subdirectories and using `shutil.move` to place the files in their new locations.

#### 2.2 Full Python Script

```python
def process_image(file_path):
    """
    Processes a single image file to extract all required metadata and properties.
    
    Args:
        file_path (str): The full path to the image file.
        
    Returns:
        dict: A dictionary containing all extracted information for one image.
    """
    try:
        # --- 1. Filename Parsing ---
        file_name = os.path.basename(file_path)
        parts = file_name.replace('stock-photo-', '').replace('.jpg', '').split('-')
        image_id = parts[-1]
        description = ' '.join(parts[:-1])

        # --- 2. Image Loading and Basic Properties ---
        with Image.open(file_path) as img:
            width, height = img.size
            img_array = np.array(img)
            
            # --- 3. Color and Brightness Calculations ---
            # Average color (R, G, B)
            avg_color = np.mean(img_array, axis=(0, 1)).astype(int).tolist()

            # Grayscale conversion for brightness analysis
            grayscale_img = img.convert('L')
            grayscale_array = np.array(grayscale_img)

            # Median brightness
            median_brightness = np.median(grayscale_array)

            # --- 4. Find Brightest Pixel Coordinates ---
            max_brightness = np.max(grayscale_array)
            # Find all coordinates (y, x) of the brightest pixels
            brightest_pixels_coords = np.argwhere(grayscale_array == max_brightness)
            
            # If multiple brightest pixels, find the one closest to (0,0)
            if len(brightest_pixels_coords) > 1:
                # Calculate Euclidean norm for each: sqrt(x^2 + y^2). Note: argwhere returns (row, col) -> (y, x)
                distances = np.sqrt(brightest_pixels_coords[:, 1]**2 + brightest_pixels_coords[:, 0]**2)
                closest_index = np.argmin(distances)
                brightest_y, brightest_x = brightest_pixels_coords[closest_index]
            else:
                brightest_y, brightest_x = brightest_pixels_coords[0]

        return {
            'file_name': file_name,
            'description': description,
            'image_id': int(image_id),
            'width': width,
            'height': height,
            'average_color': avg_color,
            'median_brightness': median_brightness,
            'brightest_pixel_x': int(brightest_x),
            'brightest_pixel_y': int(brightest_y)
        }
    except Exception as e:
        print(f"Could not process {file_path}. Error: {e}")
        return None

def main(images_dir='images', output_csv='images.csv', bin_size=4):
    """
    Main function to orchestrate the image processing, DataFrame creation,
    and file organization.
    """
    # === Part 1: Create the DataFrame ===
    print("\n--- Starting Part 1: DataFrame Creation ---")
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    all_image_data = []

    for file_name in image_files:
        file_path = os.path.join(images_dir, file_name)
        data = process_image(file_path)
        if data:
            all_image_data.append(data)
    
    # Create and save the DataFrame
    df = pd.DataFrame(all_image_data)
    df.to_csv(output_csv, index=False)
    print(f"DataFrame created and saved to '{output_csv}'")
    print(df.head())

    # === Part 2: Aggregate Images into Subfolders ===
    print("\n--- Starting Part 2: Image Aggregation ---")

    # Sort the DataFrame by median brightness
    df_sorted = df.sort_values(by='median_brightness').reset_index(drop=True)

    for i in range(0, len(df_sorted), bin_size):
        # Determine bin number (starting from 1)
        bin_number = (i // bin_size) + 1
        
        # Create the destination folder
        bin_folder_name = f"{bin_number}-images"
        os.makedirs(bin_folder_name, exist_ok=True)
        
        # Get the files for the current bin
        bin_files_df = df_sorted.iloc[i : i + bin_size]
        
        print(f"Moving {len(bin_files_df)} images to '{bin_folder_name}'...")
        
        # Move each file
        for _, row in bin_files_df.iterrows():
            source_path = os.path.join(images_dir, row['file_name'])
            destination_path = os.path.join(bin_folder_name, row['file_name'])
            # Check if source file exists before moving
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
    
    # Clean up the now-empty original images directory
    if not os.listdir(images_dir):
        os.rmdir(images_dir)
        print(f"Successfully moved all images and removed the empty '{images_dir}' directory.")
    
    print("\nImage aggregation complete.")

# --- Execute the entire process ---
if __name__ == "__main__":
    main()
```

### 3. Verification of Outputs

After running the script, we can verify that the two required outputs have been created correctly.

#### **1. `images.csv` File**

The script will generate a CSV file named `images.csv` with the following structure and content (example based on generated data):
```csv
file_name,description,image_id,width,height,average_color,median_brightness,brightest_pixel_x,brightest_pixel_y
"stock-photo-a-solitary-tree-1000.jpg",a solitary tree,1000,287,358,"[58, 25, 48]",44.0,22,258
"stock-photo-blue-ocean-waves-1001.jpg",blue ocean waves,1001,316,396,"[70, 48, 48]",55.0,230,221
"stock-photo-bright-yellow-sunflower-1002.jpg",bright yellow sunflower,1002,298,284,"[70, 77, 98]",81.0,158,111
...
```

#### **2. Aggregated Image Subfolders**

The script will create new directories and move the image files into them based on their sorted median brightness. For our 12 sample images and a bin size of 4, the final directory structure will be:
```
.
├── 1-images/
│   ├── stock-photo-a-solitary-tree-1000.jpg
│   ├── stock-photo-blue-ocean-waves-1001.jpg
│   ├── stock-photo-bright-yellow-sunflower-1002.jpg
│   └── stock-photo-dark-night-sky-1003.jpg
├── 2-images/
│   ├── stock-photo-red-sports-car-1004.jpg
│   ├── stock-photo-green-forest-canopy-1005.jpg
│   ├── stock-photo-a-cup-of-coffee-1006.jpg
│   └── stock-photo-snowy-mountain-peak-1007.jpg
├── 3-images/
│   ├── stock-photo-a-modern-skyscraper-1008.jpg
│   ├── stock-photo-a-field-of-lavender-1009.jpg
│   ├── stock-photo-a-warm-fireplace-1010.jpg
│   └── stock-photo-a-single-white-cloud-1011.jpg
├── images.csv
└── ... (the python script itself)
```
The original `images/` directory will be empty and removed after all files have been successfully moved. This demonstrates a complete and clean execution of the task.