import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

class StructuredModel:
    def __init__(self):
        self.dataframe = None
        self.train = None
        self.test = None
        self.val = None
        self.example_batch = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model = None
        self.feature_layer = None

    def execute(self):
        self.data_download()
        self.data_preparation()
        self.data_preprocessing()
        self.model_creation()
        self.model_train()
        self.model_eval()


    def data_download(self):
        url = 'https://storage.googleapis.com/applied-dl/heart.csv'
        self.dataframe = pd.read_csv(url)
        print(self.dataframe.head())

    def data_preparation(self):
        self.train, self.test = train_test_split(self.dataframe, test_size=0.2)
        self.train, self.val = train_test_split(self.train, test_size=0.2)
        print(len(self.train), '훈련 샘플')
        print(len(self.val), '검증 샘플')
        print(len(self.test), '테스트 샘플')

    def data_preprocessing(self):
        """
        batch_size = 5  # 예제를 위해 작은 배치 크기를 사용합니다.
        train_ds = self.df_to_dataset(self.train, batch_size=batch_size)
        val_ds = self.df_to_dataset(self.val, shuffle=False, batch_size=batch_size)
        test_ds = self.df_to_dataset(self.test, shuffle=False, batch_size=batch_size)

        for feature_batch, label_batch in train_ds.take(1):
            print('전체 특성:', list(feature_batch.keys()))
            print('나이 특성의 배치:', feature_batch['age'])
            print('타깃의 배치:', label_batch)

        # 특성 열을 시험해 보기 위해 샘플 배치를 만듭니다.
        self.example_batch = next(iter(train_ds))[0]

        age = feature_column.numeric_column("age")
        self.demo(age)
        """
        feature_columns = []

        # 수치형 열
        for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
            feature_columns.append(feature_column.numeric_column(header))

        # 버킷형 열
        age = feature_column.numeric_column("age")
        age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        feature_columns.append(age_buckets)

        # 범주형 열
        thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
        thal_one_hot = feature_column.indicator_column(thal)
        feature_columns.append(thal_one_hot)

        # 임베딩 열
        thal_embedding = feature_column.embedding_column(thal, dimension=8)
        feature_columns.append(thal_embedding)

        # 교차 특성 열
        crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
        crossed_feature = feature_column.indicator_column(crossed_feature)
        feature_columns.append(crossed_feature)

        self.feature_layer = layers.DenseFeatures(feature_columns)

        batch_size = 32
        self.train_ds = self.df_to_dataset(self.train, batch_size=batch_size)
        self.val_ds = self.df_to_dataset(self.val, shuffle=False, batch_size=batch_size)
        self.test_ds = self.df_to_dataset(self.test, shuffle=False, batch_size=batch_size)

    # 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
    def df_to_dataset(self, df, shuffle=True, batch_size=32):
        df = df.copy()
        labels = df.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(batch_size)

        return ds

    # 특성 열을 만들고 배치 데이터를 변환하는 함수
    def demo(self, feature_column):
        feature_layer = layers.DenseFeatures(feature_column)
        print(feature_layer(self.example_batch).numpy())

    def model_creation(self):
        self.model = tf.keras.Sequential([
            self.feature_layer,
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def model_train(self):
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=5)

    def model_eval(self):
        loss, accuracy = self.model.evaluate(self.test_ds)
        print("정확도", accuracy)

if __name__ == '__main__':
    m = StructuredModel()
    m.execute()


