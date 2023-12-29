import numpy as np
import cv2


def apply_motion_blur(image_path: str, save_path: str, degree: int = 10, angle: int = 45) -> np.ndarray:
    """
    Applies a motion blur effect to an image and saves the result.

    This function reads an image from a specified path, applies a motion blur effect characterized
    by a given degree and angle, and then saves the blurred image to another path. The function
    also returns the blurred image as a NumPy array.

    Parameters:
    image_path: The path to the input image.
    save_path: The path where the blurred image will be saved.
    degree (optional): The intensity of the motion blur. Default is 10.
    angle (optional): The angle of the motion blur in degrees. Default is 45.

    Returns:
    The blurred image as a NumPy array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"The image at {image_path} could not be found or opened")

    # Create motion blur kernel
    rotation_mat = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, rotation_mat, (degree, degree))
    motion_blur_kernel /= degree

    # Apply the motion blur kernel
    blurred_image = cv2.filter2D(image, -1, motion_blur_kernel)

    blurred_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(save_path, blurred_image)

    return blurred_image


def apply_gaussian_blur(image_path: str, save_path: str, blur_radius: int = 9) -> cv2.Mat:
    """
    Applies Gaussian blur to an image and saves the result.

    This function reads an image from the specified path, applies Gaussian blur with a given radius,
    and then saves the blurred image to another path. It also returns the blurred image as a cv2.Mat object.

    Parameters:
    image_path: The path to the input image.
    save_path: The path where the blurred image will be saved.
    blur_radius (optional): The radius of the Gaussian blur. Default is 9.

    Returns:
    The blurred image.

    Note: The blur radius should be a positive odd integer. If an even or non-positive value is provided,
    it will default to 9.
    """
    # Ensure the blur radius is a positive odd integer
    if blur_radius % 2 == 0 or blur_radius <= 0:
        blur_radius = 9

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"The file at {image_path} could not be found or opened.")

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_radius, blur_radius), sigmaX=0, sigmaY=0)

    cv2.imwrite(save_path, blurred_img)
    return blurred_img


def add_salt_and_pepper_noise(image_path: str, save_path: str, noise_intensity: float = 0.2) -> np.ndarray:
    """
    Adds salt and pepper noise to an image and saves the result.

    The function reads an image, applies salt and pepper noise based on the specified intensity,
    saves the processed image, and returns it.

    Parameters:
    image_path: Path to the input image.
    save_path: Path where the noisy image will be saved.
    noise_intensity: Intensity of the noise, represented as a fraction of the total pixels.
                     For example, 0.05 means 5% of the pixels will be affected.

    Returns:
    The image with salt and pepper noise added.

    Note: The noise_intensity should be a float between 0 and 1.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image file not found at {image_path}.")

    # Apply salt and pepper noise
    noisy_img = np.copy(img)
    num_salt = np.ceil(noise_intensity * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = 1

    num_pepper = np.ceil(noise_intensity * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = 0

    cv2.imwrite(save_path, noisy_img)
    return noisy_img


def add_gaussian_noise(image_path: str, save_path: str, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Adds Gaussian noise to an image and saves the result.

    The function reads an image, applies Gaussian noise with specified mean and standard deviation (sigma),
    saves the processed image, and returns it.

    Parameters:
    image_path: Path to the input image.
    save_path: Path where the noisy image will be saved.
    mean (optional): Mean of the Gaussian noise. Default is 0.
    sigma (optional): Standard deviation (sigma) of the Gaussian noise. Default is 25.

    Returns: 
    The image with Gaussian noise added.

    Note: The mean and sigma define the intensity of the Gaussian noise.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image file not found at {image_path}.")

    # Add Gaussian noise
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = cv2.add(img, gaussian_noise.astype(np.uint8))

    cv2.imwrite(save_path, noisy_img)
    return noisy_img
