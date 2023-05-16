import numpy as np
import os
import pickle
import random
from glob import glob


def get_noise(dist: str, n_samples: int, length: int) -> None:
    """
    Generate noise samples according to the specified distribution.

    Parameters:
        dist (str): The distribution of the noise. Supported options are "gaussian" and "uniform".
        n_samples (int): The number of noise samples to generate.
        length (int): The length of each noise sample.

    Returns:
        numpy.ndarray: An array of shape (n_samples, 1, 1, length) containing the generated noise samples.

    Raises:
        ValueError: If `dist` is not "gaussian" or "uniform".
    """
    if dist == "gaussian":
        return np.random.normal(0, 1, (n_samples, 1, 1, length))
    elif dist == "uniform":
        return np.random.uniform(0, 1, (n_samples, 1, 1, length))
    else:
        raise ValueError(
            "Invalid distribution. Supported options are 'gaussian' and 'uniform'."
        )


def rescale_pixel_values(arr):
    """
    Rescales the pixel values of a numpy array to the range [-1,1].

    Parameters:
        arr (ndarray): The input numpy array representing the image data.

    Returns:
        ndarray: The rescaled numpy array with pixel values adjusted to the range [-1,1].
    """
    for img in range(arr.shape[0]):
        # Rescale to [0,1].
        arr[img, :, :] = arr[img, :, :] - np.amin(arr[img, :, :])
        if np.amax(arr[img, :, :]) < 1:
            arr[img, :, :] = arr[img, :, :] * (1 / np.amax(arr[img, :, :]))
        elif np.amax(arr[img, :, :]) > 1:
            arr[img, :, :] = arr[img, :, :] / np.amax(arr[img, :, :])
        # Rescale to [-1,1].
        arr[img, :, :] = (arr[img, :, :] * 2) - 1

    return arr


def load_data(
    data_dir,
    data_type,
    batch_size,
    one_batch=True,
    rescale_imgs=True,
    rotated_train=None,
):
    """
    Loads and returns image data from the specified directory.

    Parameters:
        data_dir (str): The directory containing the image data.
        data_type (str): The type of data to load, such as "train" or "test".
        batch_size (int): The number of images to load per batch.
        one_batch (bool, optional): Specifies whether to load only one batch of images.
                                   If True (default), randomly selects 'batch_size' number of images.
                                   If False, selects a number of images divisible by 'batch_size'.
        rescale_imgs (bool, optional): Specifies whether to rescale the pixel values of the loaded images.
                                       If True (default), rescales the pixel values.
                                       If False, does not perform any rescaling.
        rotated_train (bool or None, optional): Specifies whether to load rotated training images.
                                                If None (default), loads non-rotated images.
                                                If True, loads rotated images.

    Returns:
        dict: A dictionary containing the loaded image data. The keys represent the image types ("t1" or "t2"),
              and the values are numpy arrays of the loaded images.
              The shape of each numpy array is (batch_size, output_shape, output_shape, 1)
    """
    # Identify image paths.
    basic_path = f"{data_dir}/{data_type}/"
    if data_type == "train":
        if rotated_train:
            basic_path = f"{basic_path}/rotated/"
        else:
            basic_path = f"{basic_path}/non_rotated/"
    all_t1_paths = glob(f"{basic_path}/t1/*")
    all_t2_paths = glob(f"{basic_path}/t2/*")

    # Identify random indices for sampling.
    if one_batch:
        indices = random.sample(range(0, len(all_t1_paths)), batch_size)
    else:
        n_all = len(all_t1_paths)
        n_return = int(n_all // batch_size) * batch_size
        indices = random.sample(range(0, len(all_t1_paths)), n_return)

    # Load and return images.
    data_dict = {}
    for t1 in [True, False]:
        if t1:
            paths = all_t1_paths
            img_type = "t1"
        else:
            paths = all_t2_paths
            img_type = "t2"

        imgs = [np.load(paths[img_idx]) for img_idx in indices]

        # Transform dimensions. Final returned shape is (len(indices), output_shape, output_shape, 1)
        imgs = np.expand_dims(imgs, 3)
        imgs_arr = np.array(imgs)

        # Rescale images.
        if rescale_imgs:
            imgs_arr = rescale_pixel_values(imgs_arr)

        data_dict[img_type] = imgs_arr

    return data_dict


def compute_total_loss(loss_target, loss_fake, mean=True):
    """
    Computes the total loss given the target loss and the fake loss.

    Parameters:
        loss_target (float): The target loss.
        loss_fake (float): The fake loss.
        mean (bool, optional): Specifies whether to compute the mean of the losses.
                               If True (default), the total loss is the sum of the target and fake losses.
                               If False, the total loss is the average of the target and fake losses.

    Returns:
        float: The computed total loss.
    """
    if mean:
        return loss_target + loss_fake
    else:
        return (loss_target + loss_fake) / 2


def write_logs_to_file(dict_losses, res_dir, filename):
    """
    Writes a dictionary of losses to a file.

    The losses dictionary is saved as a pickle file in the specified directory with the given filename.

    Parameters:
        dict_losses (dict): A dictionary containing the losses to be saved.
        res_dir (str): The directory path where the file should be saved.
        filename (str): The name of the file to be created (without the file extension).

    Returns:
        None
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    pickle.dump(
        dict_losses, open(f"{res_dir}/{filename}.pkl", "wb"), pickle.HIGHEST_PROTOCOL
    )
