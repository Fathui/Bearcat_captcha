# 函数丢这里
import re
import os
import time
import shutil
import base64
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from loguru import logger
import matplotlib.pyplot as plt
from settings import validation_path
from settings import test_path
from settings import IMAGE_HEIGHT
from settings import IMAGE_WIDTH
from settings import CAPTCHA_LENGTH
from settings import IMAGE_CHANNALS
from settings import CAPTCHA_CHARACTERS_LENGTH
from concurrent.futures import ThreadPoolExecutor


class Image_Processing(object):
    # 分割数据集
    @classmethod
    def move_path(self, path: list, proportion=0.2) -> bool:
        logger.debug(f'数据集有{len(path)},{proportion * 100}%作为验证集,{proportion * 100}%作为测试集')
        division_number = int(len(path) * proportion)
        logger.debug(f'验证集数量为{division_number},测试集数量为{division_number}')
        validation_dataset = random.sample(path, division_number)
        for i in tqdm(validation_dataset, desc='准备移动'):
            with ThreadPoolExecutor(max_workers=500) as t:
                t.submit(path.remove, i)
            # logger.debug(f'剩余{len(path)}')
        validation = [os.path.join(validation_path, os.path.split(i)[-1]) for i in validation_dataset]
        logger.debug(validation)
        for full_path, des_path in tqdm(zip(validation_dataset, validation), desc='正在移动到验证集'):
            with ThreadPoolExecutor(max_workers=500) as t:
                t.submit(shutil.move, full_path, des_path)
        test_dataset = random.sample(path, division_number)
        test = [os.path.join(test_path, os.path.split(i)[-1]) for i in test_dataset]
        for full_path, des_path in tqdm(zip(test_dataset, test), desc='正在移动到测试集'):
            with ThreadPoolExecutor(max_workers=500) as t:
                t.submit(shutil.move, full_path, des_path)
        logger.info(f'任务结束')
        return True

    # 修改文件名
    @classmethod
    def rename_path(self, path: list, original='.', reform='_'):
        for i in tqdm(path, desc='正在改名'):
            paths, name = os.path.split(i)
            name, mix = os.path.splitext(name)
            if original in name:
                new_name = name.replace(original, reform)
                os.rename(i, os.path.join(paths, new_name + mix))

    @classmethod
    def rename_suffix(self, path: list):
        for i in tqdm(path, desc='正在修改后缀'):
            paths, name = os.path.split(i)
            name, mix = os.path.splitext(name)
            os.rename(i, os.path.join(paths, name + '.jpg'))

    # 提取分类列表
    @classmethod
    def extraction_dict(self, path_list: list, divide='_') -> dict:
        paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
        path = [re.split(divide, i)[0] for i in paths]
        dicts = sorted(set(path))
        return dict((name, index) for index, name in enumerate(dicts))

    @classmethod
    # 提取全部图片plus
    def extraction_image(self, path: str) -> list:
        try:
            data_path = []
            datas = [os.path.join(path, i) for i in os.listdir(path)]
            for data in datas:
                data_path = data_path + [os.path.join(data, i) for i in os.listdir(data)]
            return data_path
        except:
            return [os.path.join(path, i) for i in os.listdir(path)]

    # 增强图片
    @classmethod
    def preprosess_save_images(self, image_path, size):
        logger.debug(f'开始处理{image_path}')
        image_name = os.path.splitext(os.path.split(image_path)[-1])[0]
        image_suffix = os.path.splitext(os.path.split(image_path)[-1])[-1]
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        # img_tensor_up = tf.image.flip_up_down(img_tensor)
        # img_tensor_a = tf.image.resize(img_tensor, size)
        # 旋转
        img_tensor_rotated_90 = tf.image.resize(tf.image.rot90(img_tensor), size)
        img_tensor_rotated_180 = tf.image.resize(tf.image.rot90(tf.image.rot90(img_tensor)), size)
        img_tensor_rotated_270 = tf.image.resize(tf.image.rot90(tf.image.rot90(tf.image.rot90(img_tensor))), size)
        # 对比度
        img_tensor_contrast1 = tf.image.resize(tf.image.adjust_contrast(img_tensor, 1), size)

        img_tensor_contrast9 = tf.image.resize(tf.image.adjust_contrast(img_tensor, 9), size)
        # 饱和度
        img_tensor_saturated_1 = tf.image.resize(tf.image.adjust_saturation(img_tensor, 1), size)

        img_tensor_saturated_9 = tf.image.resize(tf.image.adjust_saturation(img_tensor, 9), size)
        # 亮度
        img_tensor_brightness_1 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.1), size)

        img_tensor_brightness_4 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.4), size)
        # img_tensor_brightness_5 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.5), size)
        # img_tensor_brightness_6 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.6), size)
        # img_tensor_brightness_7 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.7), size)
        # img_tensor_brightness_8 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.8), size)
        # img_tensor_brightness_9 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.9), size)
        # 裁剪
        # img_tensor_crop1 = tf.image.resize(tf.image.central_crop(img_tensor, 0.1), size)
        # img_tensor_crop2 = tf.image.resize(tf.image.central_crop(img_tensor, 0.2), size)
        # img_tensor_crop3 = tf.image.resize(tf.image.central_crop(img_tensor, 0.3), size)
        # img_tensor_crop4 = tf.image.resize(tf.image.central_crop(img_tensor, 0.4), size)
        # img_tensor_crop5 = tf.image.resize(tf.image.central_crop(img_tensor, 0.5), size)
        # 调整色相
        img_tensor_hue1 = tf.image.resize(tf.image.adjust_hue(img_tensor, 0.1), size)

        img_tensor_hue9 = tf.image.resize(tf.image.adjust_hue(img_tensor, 0.9), size)
        # 图片标准化
        img_tensor_standardization = tf.image.resize(tf.image.per_image_standardization(img_tensor), size)
        # img_tensor = tf.cast(img_tensor, tf.float32)
        # img_tensor = img_tensor / 255
        image_tensor = [img_tensor_rotated_90, img_tensor_rotated_180, img_tensor_rotated_270, img_tensor_contrast1,
                        img_tensor_contrast9, img_tensor_saturated_1, img_tensor_saturated_9, img_tensor_brightness_1,
                        img_tensor_brightness_4, img_tensor_hue1, img_tensor_hue9, img_tensor_standardization]
        for index, i in tqdm(enumerate(image_tensor), desc='正在生成图片'):
            img_tensor = np.asarray(i.numpy(), dtype='uint8')
            img_tensor = tf.image.encode_jpeg(img_tensor)
            with tf.io.gfile.GFile(f'train_enhance_dataset/{image_name}_{str(index)}{image_suffix}', 'wb') as file:
                file.write(img_tensor.numpy())
        logger.info(f'处理完成{image_path}')
        return True

    @classmethod
    # 展示图片处理后的效果
    def show_image(self, image_path):
        '''
        展示图片处理后的效果
        :param image_path:
        :return:
        '''
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = np.asarray(img_tensor.numpy(), dtype='uint8')
        print(img_tensor.shape)
        print(img_tensor.dtype)
        plt.imshow(img_tensor)
        plt.show()

    @classmethod
    # 对图片进行解码,预测
    def load_image(self, path):
        '''
        预处理图片函数
        :param path:图片路径
        :return: 处理好的路径
        '''
        img_raw = tf.io.read_file(path)
        # channel 是彩色图片
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.
        img_tensor = tf.expand_dims(img_tensor, 0)
        return img_tensor

    @classmethod
    def char2pos(self, c):
        c = str(c)
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    # 文本转向量
    @classmethod
    def text2vector(self, text):
        if len(text) < CAPTCHA_LENGTH:
            while True:
                text = text + '_'
                if len(text) == CAPTCHA_LENGTH:
                    break
                else:
                    continue
        if len(text) > CAPTCHA_LENGTH:
            raise ValueError(f'有验证码长度大于{CAPTCHA_LENGTH}标签为:{text}')
        # 10个数字，大小写字母26,一个_表示不足CAPTCHA_LENGTH
        vector = np.zeros(CAPTCHA_LENGTH * CAPTCHA_CHARACTERS_LENGTH)
        for i, c in enumerate(text):
            index = i * CAPTCHA_CHARACTERS_LENGTH + self.char2pos(c)
            vector[index] = 1
        return vector

    @classmethod
    def filename2label(self, path: list):
        labels = []
        for label in path:
            tmp = []
            for letter in label:
                tmp.append(self.char2pos(letter))
            labels.append(tmp)
        return labels

    # 转换成独热编码
    @classmethod
    def extraction_one_hot_lable(self, path_list: list, suffix=True, divide='_') -> list:
        if suffix:
            lable_list = [re.split(divide, os.path.splitext(os.path.split(i)[-1])[0])[0] for i in
                          tqdm(path_list, desc='正在获取文件名')]
            # lable_list = self.filename2label(lable_list)
            # logger.error(lable_list)
            lable_list = [self.text2vector(i) for i in tqdm(lable_list, desc='正在生成numpy')]
            return lable_list
        else:
            lable_list = [os.path.splitext(os.path.split(i)[-1])[0] for i in tqdm(path_list, desc='正在获取文件名')]
            # lable_list = self.filename2label(lable_list)
            lable_list = [self.text2vector(i) for i in tqdm(lable_list, desc='正在生成numpy')]
            return lable_list

    # 向量转文本
    @classmethod
    def vector2text(self, vector):
        char_pos = vector.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % 63
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)


# 打包数据
class WriteTFRecord(object):
    @classmethod
    def WriteTFRecord(self, TFRecord_path, datasets: list, lables: list, file_name='dataset.tfrecords'):
        num_count = len(datasets)
        lables_count = len(lables)
        if not os.path.exists(TFRecord_path):
            os.mkdir(TFRecord_path)
        logger.info(f'文件个数为:{num_count}')
        logger.info(f'标签个数为:{lables_count}')
        filename = os.path.join(TFRecord_path, file_name)
        writer = tf.io.TFRecordWriter(filename)
        logger.info(f'开始保存{filename}')
        for dataset, lable in zip(datasets, lables):
            image_bytes = open(dataset, 'rb').read()
            num_count = num_count - 1
            logger.debug(f'剩余{num_count}图片待打包')
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                             'lable': tf.train.Feature(int64_list=tf.train.Int64List(value=[lable]))}))
            # 序列化
            serialized = example.SerializeToString()
            writer.write(serialized)
        logger.info(f'保存{filename}成功')
        writer.close()
        return filename

    @classmethod
    def WriteTFRecord_verification(self, TFRecord_path, datasets: list, lables: list, file_name='dataset.tfrecords'):
        num_count = len(datasets)
        lables_count = len(lables)
        if not os.path.exists(TFRecord_path):
            os.mkdir(TFRecord_path)
        logger.info(f'文件个数为:{num_count}')
        logger.info(f'标签个数为:{lables_count}')
        filename = os.path.join(TFRecord_path, file_name)
        writer = tf.io.TFRecordWriter(filename)
        logger.info(f'开始保存{filename}')
        for dataset, lable in zip(datasets, lables):
            image_bytes = open(dataset, 'rb').read()
            num_count = num_count - 1
            logger.debug(f'剩余{num_count}图片待打包')
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                             'lable': tf.train.Feature(float_list=tf.train.FloatList(value=lable))}))
            # 序列化
            serialized = example.SerializeToString()
            writer.write(serialized)
        logger.info(f'保存{filename}成功')
        writer.close()
        return filename


@tf.function
# 处理图片(将图片转化成tensorflow)
def load_preprosess_image(image_path):
    '''
    处理图片
    :param image_path:一张图片的路径
    :return:tensor
    '''
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255.
    return img_tensor


@tf.function
# 映射函数
def parse_function(exam_proto):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'lable': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(exam_proto, features)
    img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = img_tensor / 255.
    lable_tensor = parsed_example['lable']
    return (img_tensor, lable_tensor)


@tf.function
# 映射函数
def parse_function_verification(exam_proto):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'lable': tf.io.FixedLenFeature([CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH], tf.float32)
    }
    parsed_example = tf.io.parse_single_example(exam_proto, features)
    img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = img_tensor / 255.
    lable_tensor = parsed_example['lable']
    logger.debug(lable_tensor)
    return (img_tensor, lable_tensor)


class Distinguish_image(object):
    true_value = 0
    predicted_value = 0

    @classmethod
    def char2pos(self, c):
        c = str(c)
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    @classmethod
    def text2vector(self, text):
        if len(text) < CAPTCHA_LENGTH:
            while True:
                text = text + '_'
                if len(text) == CAPTCHA_LENGTH:
                    break
                else:
                    continue
        if len(text) > CAPTCHA_LENGTH:
            raise ValueError(f'有验证码长度大于{CAPTCHA_LENGTH}标签为:{text}')
        # 10个数字，大小写字母26,一个_表示不足CAPTCHA_LENGTH
        vector = np.zeros(CAPTCHA_LENGTH * CAPTCHA_CHARACTERS_LENGTH)
        for i, c in enumerate(text):
            index = i * CAPTCHA_CHARACTERS_LENGTH + self.char2pos(c)
            vector[index] = 1
        return vector

    @classmethod
    def vector2text(self, vector):
        char_pos = tf.argmax(vector, axis=1)
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % 63
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    @classmethod
    def extraction_one_hot_lable(self, path, suffix=True, divide='_'):
        if suffix:
            lable_list = re.split(divide, os.path.splitext(os.path.split(path)[-1])[0])[0]
            return lable_list
        else:
            lable_list = os.path.splitext(os.path.split(path)[-1])[0]
            return lable_list

    # 预测方法
    @classmethod
    def distinguish_image(self, model_path, jpg_path, suffix=True, divide='_'):
        model = tf.keras.models.load_model(model_path)
        forecast = model.predict(Image_Processing.load_image(jpg_path))
        lable_forecast = self.vector2text(forecast[0])
        lable_real = self.extraction_one_hot_lable(jpg_path, suffix, divide)
        logger.info(f'预测值为{lable_forecast.replace("_", "")},真实值为{lable_real.replace("_", "")}')
        if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
            logger.error(f'预测失败的图片路径为:{jpg_path}')
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
        else:
            self.predicted_value = self.predicted_value + 1
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
        return lable_forecast

    @classmethod
    # 对图片进行解码,预测
    def load_image(self, img_raw):
        '''
        预处理图片函数
        :param path:图片路径
        :return: 处理好的路径
        '''
        # img_raw = tf.io.decode_jpeg(path)
        # channel 是彩色图片
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.
        img_tensor = tf.expand_dims(img_tensor, 0)
        return img_tensor

    # 后端
    @classmethod
    def distinguish_api(self, model_path, base64_str):
        '''
        with open(file_name,'rb') as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')
        :param model_path:
        :param base64_str:
        :return:
        '''
        jpg = base64.b64decode(base64_str)
        model = tf.keras.models.load_model(model_path)
        forecast = model.predict(Distinguish_image.load_image(jpg))
        lable_forecast = self.vector2text(forecast[0])
        return lable_forecast.replace("_", "")


class AHNU(object):
    def load_img(self, image_file):
        '''
        加载图片
        '''
        image = tf.io.read_file(image_file)
        # image = tf.image.decode_image(image, channels=0)    # 多通道
        image = tf.image.decode_image(image, channels=IMAGE_CHANNALS)  # 单通道
        image = tf.cast(image, tf.float32)
        return image

    def normalize(self, image):
        '''
        像素值归一化到 -1~1
        '''
        image = (image / 127.5) - 1

        return image

    def load_image_train(self, imgs, labels):
        '''
        加载一批次训练数据
        '''
        image = self.load_img(imgs)
        image = self.normalize(image)
        image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        label = tf.strings.bytes_split(labels)
        label = tf.strings.to_number(label, tf.int32)
        label = tf.one_hot(label, depth=CAPTCHA_CHARACTERS_LENGTH)
        label = tf.cast(label, tf.float32)
        return image, label

    # 定义模型函数，更深会更慢，经测试四层卷积已能达到100%准确率
    @classmethod
    def make_model(self):
        input_size = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)
        input_layer = tf.keras.layers.Input(shape=input_size)
        conv_1_1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same')(input_layer)
        bn_1_1 = tf.keras.layers.BatchNormalization()(conv_1_1)
        relu_1_1 = tf.keras.layers.LeakyReLU()(bn_1_1)
        conv_1_2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same')(relu_1_1)
        bn_1_2 = tf.keras.layers.BatchNormalization()(conv_1_2)
        relu_1_2 = tf.keras.layers.LeakyReLU()(bn_1_2)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_1_2)
        conv_2_1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(conv_out)
        bn_2_1 = tf.keras.layers.BatchNormalization()(conv_2_1)
        relu_2_1 = tf.keras.layers.LeakyReLU()(bn_2_1)
        conv_2_2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(relu_2_1)
        bn_2_2 = tf.keras.layers.BatchNormalization()(conv_2_2)
        relu_2_2 = tf.keras.layers.LeakyReLU()(bn_2_2)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_2_2)

        '''
        conv_3_1 = layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(conv_out)
        bn_3_1 = layers.BatchNormalization()(conv_3_1)
        relu_3_1 = layers.LeakyReLU()(bn_3_1)
        conv_3_2 = layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same')(relu_3_1)
        bn_3_2 = layers.BatchNormalization()(conv_3_2)
        relu_3_2 = layers.LeakyReLU()(bn_3_2)
        conv_out = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_3_2)
        '''

        flatten = tf.keras.layers.Flatten()(conv_out)
        fc1 = tf.keras.layers.Dense(128)(flatten)
        bn_4 = tf.keras.layers.BatchNormalization()(fc1)
        relu_4_1 = tf.keras.layers.LeakyReLU()(bn_4)

        # 全连接层，输出通道数为分类数
        final_fc = tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH)(relu_4_1)

        outputs_logits = tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH))(final_fc)

        outputs = tf.keras.layers.Softmax()(outputs_logits)

        # 定义模型，指定输入与输出
        model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])
        return model

    @classmethod
    def simple_model(self):
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
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        return model


@tf.autograph.experimental.do_not_convert
def load_image_train_wrapper(imgs, labels):
    result_tensors = tf.py_function(AHNU().load_image_train, [imgs, labels], [tf.float32, tf.float32])
    result_tensors[0].set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    result_tensors[1].set_shape([CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH])
    return result_tensors


def cheak_path(path):
    while True:
        if os.path.exists(path):
            paths, name = os.path.split(path)
            name, mix = os.path.splitext(name)
            name = name + f'_{int(time.time())}'
            path = os.path.join(paths, name + mix)
        if not os.path.exists(path):
            return path
