import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CnnModel:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.model = None

    def execute(self):
        self.download_data()
        self.create_model()
        self.train_model()
        self.eval_model()

    def download_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.mnist.load_data()

        self.train_images = self.train_images.reshape((60000, 28, 28, 1))
        self.test_images = self.test_images.reshape((10000, 28, 28, 1))

        # 픽셀 값을 0~1 사이로 정규화합니다.
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        print(self.model.summary())

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def eval_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print(test_acc)

if __name__ == '__main__':
    c = CnnModel()
    c.execute()