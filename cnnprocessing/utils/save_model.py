from keras.models import model_from_json
import os


def save_model(model, path, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, model_name + '.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, model_name + '.h5'))
    print("Saved model to disk")

    return


def load_model(path, model_name):
    # load json and create model
    json_file = open(os.path.join(path, model_name + '.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(path, model_name + '.h5'))
    print("Loaded model from disk")

    return loaded_model
