import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the specified path.
    Returns None if the image could not be loaded.

    Parameters:
    image_path: The path to the input image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: {image_path} is not existed.")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Error: Image at {image_path} could not be read.")
    return image


def resize_to_fit(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Resizes the foreground image to fit within the background.

    Parameters:
    foreground: foreground image
    background: background image
    """
    scale_factor = min(background.shape[0] / foreground.shape[0],
                       background.shape[1] / foreground.shape[1])
    new_dim = (int(foreground.shape[1] * scale_factor),
               int(foreground.shape[0] * scale_factor))
    return cv2.resize(foreground, new_dim, interpolation=cv2.INTER_AREA)


def overlay_rotated_foreground(foreground_path: str, background_path: str, output_path: str) -> None:
    """
    Overlays a rotated version of the foreground image onto the background image.
    Saves the result to the specified output path.
    
    Parameters:
    foreground_path: specified path of foreground image
    background_path: specified path of background image
    output_path: specified path to save output image
    """
    foreground = load_image(foreground_path)
    background = load_image(background_path)

    # Image validity check
    if foreground.shape[0] > background.shape[0] or foreground.shape[1] > background.shape[1]:
        foreground = resize_to_fit(foreground, background)
        # return

    # Preparing foreground with Alpha channel
    rows, cols, _ = foreground.shape
    foreground_with_alpha = np.zeros((rows, cols, 4), dtype=np.uint8)
    foreground_with_alpha[:, :, :3] = foreground  # Copy RGB channels
    foreground_with_alpha[:, :, 3] = 255  # Set alpha channel to maximum

    # Create a larger canvas for rotation
    max_dim = int(np.ceil(np.sqrt(rows ** 2 + cols ** 2)))
    larger_canvas = np.zeros((max_dim, max_dim, 4), dtype=np.uint8)
    offset_x = (max_dim - cols) // 2
    offset_y = (max_dim - rows) // 2
    larger_canvas[offset_y:offset_y + rows, offset_x:offset_x + cols] = foreground_with_alpha

    # Rotate the larger canvas
    rotation_angle = random.randint(0, 360)
    center = (max_dim // 2, max_dim // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_canvas = cv2.warpAffine(larger_canvas, rotation_matrix, (max_dim, max_dim))

    # Find the bounding box of the rotated image
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int((cols * sin_theta) + (rows * cos_theta))
    new_height = int((rows * sin_theta) + (cols * cos_theta))
    new_x = int((max_dim - new_width) // 2)
    new_y = int((max_dim - new_height) // 2)

    # Extract the rotated foreground without cropping
    rotated_foreground = rotated_canvas[new_y:new_y + new_height, new_x:new_x + new_width]
    if rotated_foreground.shape[0] > background.shape[0] or rotated_foreground.shape[1] > background.shape[1]:
        rotated_foreground = resize_to_fit(rotated_foreground, background)
    new_height, new_width, _ = rotated_foreground.shape

    # Overlay the rotated foreground onto the background
    y_offset = random.randint(0, background.shape[0] - new_height)
    x_offset = random.randint(0, background.shape[1] - new_width)

    overlay_area = background[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

    # Create masks for alpha channels
    alpha_fg = rotated_foreground[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    # Blend the images (foreground over background)
    alpha_fg = np.expand_dims(alpha_fg, axis=-1)
    alpha_bg = np.expand_dims(alpha_bg, axis=-1)
    overlay_area[:, :, :3] = (alpha_fg * rotated_foreground[:, :, :3] + alpha_bg * overlay_area[:, :, :3])

    # Update the background with the overlay
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = overlay_area
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = overlay_area

    cv2.imwrite(output_path, background)


def batch_overlay_random_backgrounds(foreground_img_path: str, background_img_path: str, save_path: str,
                                     num_workers: int = 17) -> None:
    """
     Overlay each image in a batch of foreground images with a random background image.

    This function takes a directory of foreground images and a directory of background images, then overlays
    each foreground image onto a randomly selected background image. It uses multi-threading to process multiple 
    images in parallel, improving efficiency for large batches. The resulting images are saved to a specified 
    directory.

    Parameters:
    foreground_img_path: Path to the directory containing foreground images.
    background_img_path: Path to the directory containing background images.
    save_path: Path to the directory where the resultant images will be saved.
    num_workers (optional): Number of worker threads to use for parallel processing, default is 17.
    """
    os.makedirs(save_path, exist_ok=True)
    foreground_list = os.listdir(foreground_img_path)
    background_list = os.listdir(background_img_path)
    tasks = []
    for i, foreground_name in enumerate(foreground_list):
        background_name = random.choice(background_list)

        fg_path = os.path.join(foreground_img_path, foreground_name)
        bg_path = os.path.join(background_img_path, background_name)
        img_save_path = os.path.join(save_path, f"{i}.png")
        tasks.append((fg_path, bg_path, img_save_path))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda p: overlay_rotated_foreground(*p), tasks), total=len(tasks)))

