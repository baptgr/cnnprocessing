from __future__ import print_function

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import backend as K


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.

    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.

    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def _generate_filter_image(input_img,
                           layer_output,
                           filter_index,
                           step=1.,
                           epochs=15,
                           upscaling_steps=9,
                           upscaling_factor=1.2,
                           output_dim=(412, 412)
                           ):
    """Generates image for one particular filter.

    # Arguments
        input_img: The input-image Tensor.
        layer_output: The output-image Tensor.
        filter_index: The to be processed filter number.
                      Assumed to be valid.

    #Returns
        Either None if no image could be generated.
        or a tuple of the image (array) itself and the last loss.
    """
    s_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some random noise
    intermediate_dim = tuple(
        int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random(
            (1, 3, intermediate_dim[0], intermediate_dim[1]))
    else:
        input_img_data = np.random.random(
            (1, intermediate_dim[0], intermediate_dim[1], 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # Slowly upscaling towards the original size prevents
    # a dominating high-frequency of the to visualized structure
    # as it would occur if we directly compute the 412d-image.
    # Behaves as a better starting point for each following dimension
    # and therefore avoids poor local minima
    for up in reversed(range(upscaling_steps)):
        # we run gradient ascent for e.g. 20 steps
        for _ in range(epochs):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # some filters get stuck to 0, we can skip them
            if loss_value <= K.epsilon():
                return None

        # Calculate upscaled dimension
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** up)) for x in output_dim)
        # Upscale
        img = deprocess_image(input_img_data[0])
        img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
                                                       pil_image.BICUBIC))
        input_img_data = np.expand_dims(
            process_image(img, input_img_data[0]), 0)

    # decode the resulting input image
    img = deprocess_image(input_img_data[0])
    e_time = time.time()
    print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                              loss_value,
                                                              e_time - s_time))
    return img, loss_value


def _draw_filters(img_loss, output_dim, layer_name, filter_index):
    """Draw the best filters in a nxn grid.

    # Arguments
        filters: A List of generated images and their corresponding losses
                 for each processed filter.
        n: dimension of the grid.
           If none, the largest possible square will be used
    """


    # build a black picture with enough space for
    # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
    width = output_dim[0]
    height = output_dim[1]
    stitched_filters = np.zeros((width, height, 3), dtype='uint8')

    # fill the picture with our saved filters

    img, _ = img_loss
    width_margin = output_dim[0]
    height_margin = output_dim[1]
    stitched_filters[
    width_margin: width_margin + output_dim[0],
    height_margin: height_margin + output_dim[1], :] = img


    # save the result to disk
    save_img('vgg_{0:}_{1:}.png'.format(layer_name, img), stitched_filters)