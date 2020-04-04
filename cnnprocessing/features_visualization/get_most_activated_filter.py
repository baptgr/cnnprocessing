from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt

from cnnprocessing.features_visualization.generate_filter import visualize_filter


def create_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers]
    # Extracts the outputs of the top 12 layers
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # Creates a model that will return these outputs, given the model input
    # https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

    return activation_model


def get_most_activated_filters(img, activation_model, n_filters=4, plot=True):
    layer_outputs = [layer.output for layer in activation_model.layers]

    activations = activation_model.predict(img)

    top_filters = {}

    for i in range(len(activations)):

        if layer_outputs[i].name.split('_')[1][:4] == 'conv':
            print('-------- ', layer_outputs[i].name, ' --------')

            # Take mean activation from each filter of the layer
            mean_activation = np.mean(activations[i], axis=(0, 1, 2))

            # Keep most activated filters
            top_filters[layer_outputs[i].name.split('/')[0]] = mean_activation.argsort()[-n_filters:][::-1]

            if plot:
                plt.plot(np.mean(activations[i], axis=(0, 1, 2)))
                plt.show()

    return top_filters


def generate_most_activated_filter(model, top_filers: dict, path_output_folder):

    if not os.path.exists(path_output_folder):
        os.makedirs(path_output_folder)

    for layer_name, top_filters in top_filers.items():
        print('------   ', layer_name, '   ------')
        for f in top_filters:
            visualize_filter(model, layer_name, f, path_output_folder)

        print('\n')

    return
