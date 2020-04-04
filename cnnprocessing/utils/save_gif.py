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


def save_gif_concat(path_image_input, path_images_folder, path_output, size=(412, 412), bw=True, duration=0.5):

    # Load and resize left_image
    img_input = Image.open(path_image_input)
    img_input = resize_and_crop(img_input, size)

    images = []
    for filename in os.listdir(path_images_folder):
        path_image = os.path.join(path_images_folder, filename)

        if bw:
            img = Image.open(path_image).convert('L')
        else:
            img = Image.open(path_image)

        # resize left image
        img = resize_and_crop(img, size)

        # Concat two images
        img_concat = get_concat_h(img_input, img)

        images.append(img_concat)

    imageio.mimsave(path_output, images, duration=duration)


from PIL import Image


def resize_and_crop(img, size, crop_type='middle'):
    """
    Resize and crop an image to fit the specified size.

    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'middle' or
            'bottom/right' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.

    https://gist.github.com/sigilioso/2957026
    """
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], round(size[0] * img.size[1] / img.size[0])),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, round((img.size[1] - size[1]) / 2), img.size[0],
                   round((img.size[1] + size[1]) / 2))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((round(size[1] * img.size[0] / img.size[1]), size[1]),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (round((img.size[0] - size[0]) / 2), 0,
                   round((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]),
                         Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    return img


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst