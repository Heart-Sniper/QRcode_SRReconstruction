import cv2
import numpy as np


def convert_to_transparent(img_path: str, save_path: str, color_transparent: list = [255, 255, 255]) -> np.ndarray:
    """
    Convert specified color in an image to transparent and save the modified image.

    This function reads an image from a given path, converts a specified color (defaulting to white) to transparent, 
    and then saves the modified image to a new file. 

    Parameters:
    img_path : Path to the input image file.
    save_path : Path where the modified image will be saved.
    color_transparent (optional): The RGB color to be made transparent, default is white [255, 255, 255].

    Returns:
    numpy.ndarray: The modified image with specified color converted to transparent.
    """
    img = cv2.imread(img_path)
    mask = np.all(img[:, :, :] == color_transparent, axis=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img[mask, 3] = 0

    cv2.imwrite(save_path, img)
    return img
