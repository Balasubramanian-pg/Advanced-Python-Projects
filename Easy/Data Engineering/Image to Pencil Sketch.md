---
title: Image to Pencil Sketch
company: LGM
difficulty: Easy
category: Data Engineering
date: 2025-07-28
---
_This data project has been used as an assignment during the LGM Data Science Virtual Internship._

## Assignment

Select an image in RGB format and use the `OpenCV` Python library to transform it so that it resembles a pencil sketch.

**Tips**

1. Convert the RGB image to grayscale - this will turn the image into a classic black-and-white photo;
2. Invert the grayscale image - this is sometimes referred to as a negative image and can be used to enhance details;
3. Mix the grayscale image with the inverted blurry image - this can be done by dividing the pixel values of the grayscale image by the pixel values of the inverted blurry image; the result should resemble a pencil sketch;
4. Experiment by applying other transformations offered by the `OpenCV` library to improve the effect;

## Data Description

We are providing you with a sample image of a dog, however, you can choose any colored image you want to complete this project.

## Practicalities

Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final files.

# Solution
Here is a complete, structured solution to the LGM Data Science Virtual Internship project on creating a pencil sketch effect with OpenCV.

This response is designed like a Jupyter Notebook. It includes:
1.  **Code to Generate a Sample Dataset:** Since a specific image is not required, I will first generate a synthetic sample image using Python libraries. This ensures the entire solution is fully reproducible and has predictable features (like sharp edges and color gradients) to test the effect.
2.  **A Step-by-Step Transformation Process:** The solution follows the recommended steps, explaining the purpose and showing the visual result of each transformation.
3.  **Code and Visualizations:** Each section includes the necessary code and displays the resulting images using `matplotlib`, which integrates well with programming environments like Jupyter.
4.  **Experimentation and Refinements:** The solution goes beyond the basic requirements by demonstrating how to parameterize the effect and add further enhancements, as suggested in the assignment.

***

## LGM VIP: Image to Pencil Sketch Transformation

### Project Objective
The goal of this project is to take a standard RGB color image and transform it into an artistic representation that mimics a hand-drawn pencil sketch. We will use the `OpenCV` library in Python to perform a series of image processing steps, including color space conversion, inversion, blurring, and blending.

### 1. Setup and Data Generation

First, we will import the necessary libraries. We need `OpenCV` for image processing, `NumPy` for creating our sample image array, and `Matplotlib` for displaying the images. We'll then generate a sample image to work with.

#### 1.1 Import Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set default figure size for matplotlib
plt.rcParams['figure.figsize'] = [10, 8]
```

#### 1.2 Generate a Sample Image
To ensure this project is fully reproducible, we will programmatically create a sample image named `sample_image.jpg`. This image will contain colored shapes, sharp edges, and gradients, making it a good candidate for testing our sketch effect.

```python
def generate_sample_image(file_path='sample_image.jpg', width=600, height=400):
    """
    Generates a simple colored image with geometric shapes and saves it.
    """
    # Create a blank white canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a blue and red background gradient
    for i in range(width):
        ratio = i / width
        color1 = (255 * (1-ratio), 0, 0) # Blue
        color2 = (0, 0, 255 * ratio)     # Red
        img[:, i] = [c1 + c2 for c1, c2 in zip(color1, color2)]
        
    # Draw a green circle in the center
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 100, (0, 255, 0), -1) # -1 for filled circle
    
    # Draw a yellow rectangle
    cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 255), 5) # 5 for thickness
    
    cv2.imwrite(file_path, img)
    print(f"Sample image saved to '{file_path}'")
    return file_path

# Generate and get the path of our sample image
image_path = generate_sample_image()
```

<hr>

### 2. Step-by-Step Image Transformation

Now, we will follow the process outlined in the assignment to convert the original image into a pencil sketch. We will visualize the output of each step to understand its contribution to the final effect.

#### Step 1: Load and Convert to Grayscale
The first step is to load our color image and convert it to grayscale. A grayscale image simplifies the data from three color channels (R, G, B) to a single channel representing intensity (lightness/darkness). This is the foundation of a classic black-and-white sketch.

```python
# Load the original RGB image
original_image = cv2.imread(image_path)
# OpenCV loads images in BGR format, so we convert it to RGB for correct display in matplotlib
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Convert the original image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Display the images
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('Original RGB Image')
plt.imshow(original_image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()
```

#### Step 2: Invert the Grayscale Image
Next, we invert the grayscale image. In an 8-bit image, this means subtracting each pixel's value from 255. Dark areas become light, and light areas become dark. This "negative" image is a key component for creating the sketch effect.

```python
# Invert the grayscale image
inverted_gray_image = 255 - gray_image

plt.figure()
plt.title('Inverted Grayscale Image')
plt.imshow(inverted_gray_image, cmap='gray')
plt.axis('off')
plt.show()
```

#### Step 3: Blur the Inverted Image
This is the most critical step for creating the sketch lines. We apply a Gaussian blur to the *inverted* image. The blur softens the image, spreading the light from the bright areas. The edges, which had sharp transitions, will now have smooth gradients. The size of the blur kernel (e.g., `(21, 21)`) determines the thickness of the final sketch lines.

```python
# Apply Gaussian blur to the inverted image
blurred_inverted_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)

plt.figure()
plt.title('Blurred Inverted Image')
plt.imshow(blurred_inverted_image, cmap='gray')
plt.axis('off')
plt.show()
```

#### Step 4: Mix the Grayscale and Blurred Images
Finally, we blend the original grayscale image with the blurred inverted image. The technique used here is called **Color Dodge**. It works by dividing the bottom layer (original grayscale) by the inverted top layer (blurred inverted image).

-   Where the blurred image is white (255), the result will be unchanged.
-   Where the blurred image is darker, the result will be brighter, effectively creating the sketch lines.

We use `cv2.divide()` because it handles the division gracefully and avoids divide-by-zero errors. We scale the result by 256 to get the final 8-bit image.

```python
# Invert the blurred image to use in the division
inverted_blurred_image = 255 - blurred_inverted_image

# Create the pencil sketch by dividing the grayscale image by the inverted blurred image
pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

# Display the final result
plt.figure()
plt.title('Final Pencil Sketch')
plt.imshow(pencil_sketch, cmap='gray')
plt.axis('off')
cv2.imwrite('pencil_sketch_result.jpg', pencil_sketch)
print("Final sketch saved to 'pencil_sketch_result.jpg'")
plt.show()
```

### 3. Encapsulating the Logic into a Function

To make this process reusable and to easily experiment with parameters, we can wrap the entire logic in a single function.

```python
def create_pencil_sketch(image_path, kernel_size=21):
    """
    Takes an image path and returns a pencil sketch version of it.
    
    Args:
        image_path (str): The path to the input image.
        kernel_size (int): The size of the Gaussian blur kernel (must be an odd number).
        
    Returns:
        numpy.ndarray: The resulting pencil sketch image.
    """
    # 1. Read image and convert to grayscale
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Invert the grayscale image
    inverted_gray = 255 - gray_img
    
    # 3. Blur the inverted image
    blurred_img = cv2.GaussianBlur(inverted_gray, (kernel_size, kernel_size), 0)
    
    # 4. Invert the blurred image and blend
    inverted_blurred = 255 - blurred_img
    sketch = cv2.divide(gray_img, inverted_blurred, scale=256.0)
    
    return sketch

# Test the function
final_sketch_from_func = create_pencil_sketch(image_path)

plt.figure()
plt.title('Sketch Created by Reusable Function')
plt.imshow(final_sketch_from_func, cmap='gray')
plt.axis('off')
plt.show()
```

### 4. Experimentation and Refinements

As suggested, we can experiment with other transformations to improve the effect.

#### Experiment 1: Varying the Blur Kernel Size
The `kernel_size` in the `GaussianBlur` function has a major impact on the final look. A smaller kernel results in finer, sharper lines, while a larger kernel creates a softer, more abstract sketch.

```python
# Generate sketches with different kernel sizes
sketch_fine = create_pencil_sketch(image_path, kernel_size=11)
sketch_medium = create_pencil_sketch(image_path, kernel_size=31)
sketch_soft = create_pencil_sketch(image_path, kernel_size=51)

# Display the results
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title('Fine Sketch (Kernel Size = 11)')
plt.imshow(sketch_fine, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Medium Sketch (Kernel Size = 31)')
plt.imshow(sketch_medium, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Soft Sketch (Kernel Size = 51)')
plt.imshow(sketch_soft, cmap='gray')
plt.axis('off')
plt.show()
```

#### Experiment 2: Adding a Color Sketch Effect
We can extend the technique to create a colored pencil sketch. This is done by blending the grayscale sketch we created with the original color image.

```python
def create_color_sketch(image_path, kernel_size=21):
    """
    Creates a colored pencil sketch effect.
    """
    # First, get the grayscale sketch
    gray_sketch = create_pencil_sketch(image_path, kernel_size)
    
    # Read the original color image
    color_img = cv2.imread(image_path)
    
    # Blend the color image with the grayscale sketch
    # We use cv2.cvtColor to convert the single-channel sketch to 3 channels to allow blending
    gray_sketch_3_channel = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
    
    # Use cv2.bitwise_and to blend, keeping the dark lines of the sketch
    # and the color of the original where the sketch is white
    color_sketch = cv2.bitwise_and(color_img, gray_sketch_3_channel)
    
    return color_sketch

# Create and display the color sketch
color_sketch_result = create_color_sketch(image_path)
color_sketch_rgb = cv2.cvtColor(color_sketch_result, cv2.COLOR_BGR2RGB)

plt.figure()
plt.title('Colored Pencil Sketch')
plt.imshow(color_sketch_rgb)
plt.axis('off')
cv2.imwrite('color_sketch_result.jpg', color_sketch_result)
print("Color sketch saved to 'color_sketch_result.jpg'")
plt.show()
```
### Conclusion
This project successfully demonstrates how a series of relatively simple image processing steps in OpenCV can be combined to create a compelling artistic effect. By converting an image to grayscale, inverting it, applying a blur, and then using a "Color Dodge" blending technique, we were able to transform a standard color photograph into a convincing pencil sketch. Further experimentation showed that adjusting parameters like the blur kernel size can significantly alter the artistic style, and the same core technique can be extended to produce colored sketches.