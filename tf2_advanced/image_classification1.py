from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

class CnnModel:
    def __init__(self):
        pass

    def download(self):

        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))

        # 픽셀 값을 0~1 사이로 정규화합니다.
        train_images, test_images = train_images / 255.0, test_images / 255.0

        _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

        path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

        PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
        self.train_dir = os.path.join(PATH, 'train')
        self.validation_dir = os.path.join(PATH, 'validation')
        train_cats_dir = os.path.join(self.train_dir, 'cats')  # directory with our training cat pictures
        train_dogs_dir = os.path.join(self.train_dir, 'dogs')  # directory with our training dog pictures
        validation_cats_dir = os.path.join(self.validation_dir, 'cats')  # directory with our validation cat pictures
        validation_dogs_dir = os.path.join(self.validation_dir, 'dogs')  # directory with our validation dog pictures

        num_cats_tr = len(os.listdir(train_cats_dir))
        num_dogs_tr = len(os.listdir(train_dogs_dir))

        num_cats_val = len(os.listdir(validation_cats_dir))
        num_dogs_val = len(os.listdir(validation_dogs_dir))

        self.total_train = num_cats_tr + num_dogs_tr
        self.total_val = num_cats_val + num_dogs_val
        print('total training cat images:', num_cats_tr)
        print('total training dog images:', num_dogs_tr)

        print('total validation cat images:', num_cats_val)
        print('total validation dog images:', num_dogs_val)
        print("--")
        print("Total training images:", self.total_train)
        print("Total validation images:", self.total_val)
        self.batch_size = 128
        self.epochs = 15
        self.IMG_HEIGHT = 150
        self.IMG_WIDTH = 150
    def preparation_data(self):
        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
        self.train_data_gen = train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                   directory=self.train_dir,
                                                                   shuffle=True,
                                                                   target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                   class_mode='binary')
        self.val_data_gen = validation_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                      directory=self.validation_dir,
                                                                      target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                      class_mode='binary')
        self.sample_training_images, _ = next(self.train_data_gen)

    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    def create_mode(self):
        self.model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
    def model_eval(self):
        self.history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=self.total_train // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_data_gen,
            validation_steps=self.total_val // self.batch_size
        )
    def save_model(self):
        # 전체 모델을 HDF5 파일로 저장합니다
        self.model.save('cat_dog_model.h5')

    def visualize_training_results(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    def load_model(self):
        self.new_model = keras.models.load_model('cat_dog_model.h5')
        # print(self.new_model.summary())
        # loss, acc = self.new_model.evaluate(self.test_images, self.test_labels, verbose=2)
        # print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
    def execute(self):
        self.download()
        self.preparation_data()
        self.plotImages(self.sample_training_images[:5])
        self.create_mode()
        self.model_eval()
        self.save_model()
    def execute_load(self):
        self.load_model()
# def choose_one(i):
#     while 1:
#         def print_menu()
#             print('0. EXIT\n'
#                   '1. SAVE\n'
#                   '2. LOAD\n')
#             return int(input('CHOOSE ONE\n'))
#         menu = print_menu()
#         print('MENU %s' % menu)
#         if menu == 0:
#             break
#         elif menu == 1:
#             m.



if __name__ == '__main__':
    m = CnnModel()


    m.execute()
    m.execute_load()
    # print(tf.__version__)
    # print(keras.__version__)