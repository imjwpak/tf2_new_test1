import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools

class GanModel:
    def __init__(self):
        pass
    def execute(self):
        # self.tensor_to_image()
        self.download()
        content_image = self.load_img(self.content_path)
        style_image = self.load_img(self.style_path)

        plt.subplot(1, 2, 1)
        self.imshow(content_image, 'Content Image')
        plt.subplot(1, 2, 2)
        self.imshow(style_image, 'Style Image')
        plt.show()
        import tensorflow_hub as hub
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        self.tensor_to_image(stylized_image).show()
        self.vgg19(content_image)
    def vgg19(self,content_image):
        x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
        x = tf.image.resize(x, (224, 224))
        vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        prediction_probabilities = vgg(x)
        print(prediction_probabilities.shape)

        predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
        [(class_name, prob) for (number, class_name, prob) in predicted_top_5]

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        print()
        for layer in vgg.layers:
            print(layer.name)

    def tensor_to_image(self,tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)
    def download(self):
        self.content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
                                               'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

        # https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
        self.style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                             'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    def load_img(self,path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def imshow(self,image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        plt.imshow(image)
        if title:
            plt.title(title)
if __name__ == '__main__':
    m = GanModel()
    m.execute()