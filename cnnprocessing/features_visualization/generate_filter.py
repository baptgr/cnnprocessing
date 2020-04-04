import os

from keras import layers
from cnnprocessing.features_visualization.utils import _generate_filter_image
from keras.preprocessing.image import save_img


def visualize_filter(model,
                     layer_name,
                     filter_index,
                     path_output_folder,
                     step=1.,
                     epochs=15,
                     upscaling_steps=9,
                     upscaling_factor=1.2,
                     output_dim=(412, 412)):
    """Visualizes the most relevant filters of one conv-layer in a certain model.

    # Arguments
        model: The model containing layer_name.
        layer_name: The name of the layer to be visualized.
                    Has to be a part of model.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
                         Starting image is in this case (80, 80).
        upscaling_factor: Factor to which to slowly upgrade
                          the image towards output_dim.
        output_dim: [img_width, img_height] The output image dimensions.
        filter_range: Tupel[lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`,
                      the last filter will be inferred as the upper boundary.
    """
    # this is the placeholder for the input images
    assert len(model.inputs) == 1
    input_img = model.inputs[0]

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # iterate through each filter and generate its corresponding image
    processed_filter = _generate_filter_image(input_img, output_layer.output, filter_index,
                                              step, epochs, upscaling_steps, upscaling_factor, output_dim)

    if processed_filter is not None:

        print('Filter processed')
        # Finally draw and store the best filters to disk
        save_img(os.path.join(path_output_folder, 'vgg_{0:}_{1:}.png'.format(layer_name, filter_index)), processed_filter[0].reshape((412, 412, 3)))

    else:

        print('Returns None')
