# 模型
import tensorflow as tf
from settings import IMAGE_HEIGHT
from settings import IMAGE_WIDTH
from settings import CAPTCHA_LENGTH
from settings import IMAGE_CHANNALS
from settings import CAPTCHA_CHARACTERS_LENGTH


class Model(object):

    @staticmethod
    def xception_model(fine_ture_at=3):
        covn_base = tf.keras.applications.Xception(include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                                   pooling='max')
        model = tf.keras.Sequential()
        model.add(covn_base)
        # model.add(tf.keras.layers.BatchNormalization)
        # model.add(tf.keras.layers.Dense(1024, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(512, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
        covn_base.trainable = False
        for layer in covn_base.layers[:fine_ture_at]:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])

        return model

    @staticmethod
    def simple_model():
        # input_layer = tf.keras.layers.Input(shape=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNALS))
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        return model

    @staticmethod
    def captcha_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        return model

    @staticmethod
    def captcha2_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.SeparableConv2D(16, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.SeparableConv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.SeparableConv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.SeparableConv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        return model


if __name__ == '__main__':
    # model = Model.captcha_model()
    model = Model.captcha_model()
    model.summary()
