import imageio
import os
from PIL import Image


def save_gif(path_images_folder, path_output, bw=True, duration=0.5):
    """
    This functions save images in specified folder into a GIF

    :param path_images_folder:
    :param path_output:
    :param bw:
    :param duration:
    :return:
    """
    images = []
    for filename in os.listdir(path_images_folder):
        path_image = os.path.join(path_images_folder, filename)

        if bw:
            images.append(Image.open(path_image).convert('L'))
        else:
            images.append(Image.open(path_image))

    imageio.mimsave(path_output, images, duration=duration)
    return
