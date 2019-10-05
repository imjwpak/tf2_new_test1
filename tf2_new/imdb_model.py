import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

class ImdbModel:
    def __init__(self):
        self.train_validation_split = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.model = None
        self.train_examples_batch = None


    def execute(self):
        # self.env_info()
        self.download_data() # IMDB 데이터셋 마이닝
        self.create_sample() # 샘플데이터 생성
        self.create_model() # 모델 생성
        self.train_model() # 모델 훈련
        self.eval_model() # 모델 평가

    @staticmethod
    def env_info():
        print("버전: ", tf.__version__)
        print("즉시 실행 모드: ", tf.executing_eagerly())
        print("허브 버전: ", hub.__version__)
        print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

    def download_data(self):
        # 훈련 세트를 6대 4로 나눕니다.
        # 결국 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
        self.train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

        (self.train_data, self.validation_data), self.test_data = tfds.load(
            name="imdb_reviews",
            split=(self.train_validation_split, tfds.Split.TEST),
            as_supervised=True)

    def create_sample(self):
        self.train_examples_batch, train_labels_batch = next(iter(self.train_data.batch(10)))
        print(self.train_examples_batch)
        print(train_labels_batch)

    def create_model(self):
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[],
                                   dtype=tf.string, trainable=True)
        hub_layer(self.train_examples_batch[:3])

        self.model = tf.keras.Sequential()
        self.model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train_model(self):
        history = self.model.fit(self.train_data.shuffle(10000).batch(512),
                            epochs=20,
                            validation_data=self.validation_data.batch(512),
                            verbose=1)

    def eval_model(self):
        results = self.model.evaluate(self.test_data.batch(512), verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f" % (name, value))

if __name__ == '__main__':
    m = ImdbModel()
    m.execute()