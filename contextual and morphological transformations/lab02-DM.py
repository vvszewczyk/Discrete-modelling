import numpy as np
import csv
from PIL import Image

def dilatation(image_array, radius=1):
    height, width = image_array.shape
    result = np.copy(image_array)

    for x in range(width):
        for y in range(height):
            if image_array[y, x] == 0:
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        if 0 <= y + j < height and 0 <= x + i < width:
                            result[y + j, x + i] = 0

    return result

def erosion(image_array, radius=1):
    height, width = image_array.shape
    result = np.copy(image_array)

    for x in range(width):
        for y in range(height):
            if image_array[y, x] == 255:
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        if 0 <= y + j < height and 0 <= x + i < width:
                            result[y + j, x + i] = 255

    return result


def opening(image, radius=1):
    eroded_image = erosion(image, radius)
    return dilatation(eroded_image, radius)


def closing(image, radius=1):
    dilated_image = dilatation(image, radius)
    return erosion(dilated_image, radius)


def convolution(image_array, kernel, radius):
    height, width = image_array.shape
    result = np.zeros_like(image_array)

    for x in range(width):
        for y in range(height):
            sum_value = 0
            for j in range(-radius, radius + 1):
                for i in range(-radius, radius + 1):
                    if 0 <= y + j < height and 0 <= x + i < width:
                        pixel_value = image_array[y + j, x + i]
                        kernel_value = kernel[j + radius, i + radius]
                        sum_value += pixel_value * kernel_value
            result[y, x] = sum_value

    return result


def readMaskCSV(filepath):
    mask = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            mask.append([float(value) for value in row])
    return np.array(mask)



if __name__ == "__main__":
    image_path = "images/binarized_sultan.png"
    image_pil = Image.open(image_path).convert('L')
    image = np.array(image_pil)

    dilatation_result = dilatation(image, radius=3)
    Image.fromarray(dilatation_result).save('images/dilatation_res.png')

    erosion_result = erosion(image, radius=3)
    Image.fromarray(erosion_result).save('images/erosion_res.png')

    opening_result = opening(image, radius=3)
    Image.fromarray(opening_result).save('images/opening_res.png')

    closing_result = closing(image, radius=3)
    Image.fromarray(closing_result).save('images/closing_res.png')

    image_path = "images/sultan.png"
    image_pil = Image.open(image_path).convert('L')
    image = np.array(image_pil)

    low_pass_mask = readMaskCSV('masks/low_pass_mask.csv')
    convolution_result = convolution(image, low_pass_mask, 1)
    Image.fromarray(convolution_result).save('images/low_pass_filter_res.png')

    upper_pass_mask = readMaskCSV('masks/upper_pass_mask.csv')
    convolution_result = convolution(image, upper_pass_mask, 1)
    Image.fromarray(convolution_result).save('images/upper_pass_filter_res.png')

    sevenSeven_mask = readMaskCSV('masks/7x7_mask.csv')
    convolution_result = convolution(image, sevenSeven_mask, 3)
    Image.fromarray(convolution_result).save('images/7x7_filter_res.png')

    print("Operations completed. Results saved as PNG files.")
