import os
import numpy as np
import random
from tqdm import tqdm
import cv2
import qrcode
import string
from concurrent.futures import ThreadPoolExecutor


def generate_random_string(length: int) -> str:
    """
    Generates a random string of a specified length.

    This function creates a string composed of random ASCII letters (both uppercase and lowercase) and digits.
    It's useful for generating unique identifiers or random data for testing.

    Parameters:
    length: The length of the random string to be generated.

    Returns:
    A random string of the specified length.
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def generate_qrcode(generate_num: int, save_path: str) -> None:
    """
    Generates a specified number of QR codes and saves them to a given directory.

    This function creates QR codes with random content and various sizes and saves them as PNG files.
    It's designed to generate multiple QR codes in parallel, improving performance for bulk creation.

    Parameters:
    generate_num: The number of QR codes to generate.
    save_path: The directory path where the QR code images will be saved.
               The function will create the directory if it does not exist.

    Note: Each QR code contains a randomly generated string of 10 characters as its data.
    """
    os.makedirs(save_path, exist_ok=True)

    def generate_and_save_qrcode(i):
        qr = qrcode.QRCode(version=random.randint(1, 8), box_size=random.randint(7, 13), border=0)
        data = generate_random_string(10)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="black")
        img.save(os.path.join(save_path, f"{i + 1}.png"))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(generate_and_save_qrcode, range(generate_num)), total=generate_num))


def generate_background(generate_num: int, color: tuple, shape: tuple, save_path: str, if_random=False) -> None:
    """
    Generates a set of solid color background images.

    The function creates a directory (if it doesn't exist), and then generates each image,
    either with fixed or random sizes. Images are saved in the specified directory using
    parallel processing for efficiency.

    Parameters:
    generate_num: Number of images to generate.
    color: Color (BGR format) for the images.
    shape: Dimensions of the images. Represents size range if 'if_random' is True.
    save_path: Directory to save the images.
    if_random (optional): Generates images with random sizes within the given range if True.

    """
    os.makedirs(save_path, exist_ok=True)

    def generate_single_image(i):
        if if_random:
            # Random size for the image
            width = random.randint(int(shape[0]), int(shape[1]))
            height = random.randint(int(shape[0]), int(shape[1]))
        else:
            width, height = shape
        solid_color_image = np.full((height, width, 3), color, dtype=np.uint8)

        cv2.imwrite(os.path.join(save_path, f"{i}.png"), solid_color_image)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(generate_single_image, range(generate_num)), total=generate_num))
