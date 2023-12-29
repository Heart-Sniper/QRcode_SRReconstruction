import os
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor

__all__ = ['batch_random_process']

from . import clarity_reducer as reducer


def process_image(image: str, i, save_dic: str) -> None:
    """
    Applies various processing techniques to a single image.

    Parameters:
    image: Path to the input image.
    i: Index of the image, used for naming output files.
    save_dic: Directory to save the processed images.
    """
    reducer.apply_motion_blur(image_path=image,
                              save_path=os.path.join(save_dic, f"{i}mb.png"),
                              angle=random.randint(0, 90))
    reducer.apply_gaussian_blur(image_path=image,
                                save_path=os.path.join(save_dic, f"{i}gb.png"),
                                blur_radius=random.randrange(3, 17, 2))
    reducer.add_salt_and_pepper_noise(image_path=image,
                                      save_path=os.path.join(save_dic, f"{i}spn.png"),
                                      noise_intensity=round(random.uniform(0, 0.7), 2))


def batch_random_process(image_dic: str, save_dic: str) -> None:
    """
    Processes a batch of images in parallel, applying random effects.

    Parameters:
    image_dic: Directory containing the images to process.
    save_dic: Directory to save the processed images.
    """
    os.makedirs(save_dic, exist_ok=True)
    image_list = [os.path.join(image_dic, image_name) for image_name in os.listdir(image_dic)]
    with ThreadPoolExecutor() as executor:
        image_index_pairs = [(image, i) for i, image in enumerate(image_list)]
        list(tqdm(executor.map(lambda p: process_image(p[0], p[1], save_dic), image_index_pairs), total=len(image_list)))
