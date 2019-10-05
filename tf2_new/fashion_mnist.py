import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FashionModel:
    def __init__(self):
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()

        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = None

    def execute(self):
        # ------------ 데이터셋 정보 -------------
        self.show_dataset()

        # ------------ 모델 구성 & 훈련 --------------
        self.create_model()

        # ----------- 예측하기 ---------------
        predictions = self.predict_image()

        # 테스트 : 예측한 값과 같은지 테스트 데이터의 레이블 확인
        # 테스트
        self.subplot_test(predictions)
        
        # 이미지 하나 테스트
        self.one_test()

    def show_dataset(self):
        print('-------Train set spec ----------')
        print('훈련이미지 :', self.train_images.shape)
        print('훈련이미지 수 :', len(self.train_labels))
        print('훈련이미지 라벨:', self.train_labels)
        print('-------Test set spec ----------')
        print('테스트이미지 :', self.test_images.shape)
        print('테스트이미지 수:', len(self.test_labels))
        print('테스트이미지 라벨:', self.test_labels)

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.show()

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(self.train_images, self.train_labels, epochs=5)

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\n테스트 정확도:', test_acc)

    def predict_image(self):
        predictions = self.model.predict(self.test_images)
        print('예측값 :', predictions[0])
        print('가장 신뢰도가 높은 레이블 :', np.argmax(predictions[0]))

        return predictions

    def subplot_test(self, predictions):
        print('테스트 데이터 :', self.test_labels[0])

        # method 자리
        i = 0
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        self.plot_image(i, predictions, self.test_labels, self.test_images)
        plt.subplot(1, 2, 2)
        self.plot_value_array(i, predictions, self.test_labels)
        plt.show()

        # 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
        # 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions, self.test_labels, self.test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions, self.test_labels)
        plt.show()

    def one_test(self):
        # 테스트 세트에서 이미지 하나를 선택합니다
        img = self.test_images[0]
        print(img.shape)

        # 이미지 하나만 사용할 때도 배치에 추가합니다
        img = (np.expand_dims(img, 0))
        print(img.shape)

        predictions_single = self.model.predict(img)
        print(predictions_single)

        self.plot_value_array(0, predictions_single, self.test_labels)
        _ = plt.xticks(range(10), self.class_names, rotation=45)
        print(np.argmax(predictions_single[0]))

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

if __name__ == '__main__':
    f = FashionModel()
    f.execute()