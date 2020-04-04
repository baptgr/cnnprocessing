from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


def load_image_vgg16(path):
    img = load_img(path, target_size=(224, 224))  # Charger l'image
    img = img_to_array(img)  # Convertir en tableau numpy
    img = img.reshape(
        (1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    return img
