# 测试模型
import os
import random
import tensorflow as tf
from loguru import logger
from settings import BATCH_SIZE
from settings import model_path
from settings import test_pack_path
from settings import MODEL_LEAD_NAME
from settings import test_path
from Function_API import Image_Processing
from Function_API import Distinguish_image
from Function_API import parse_function_verification

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
    parse_function_verification).batch(BATCH_SIZE)

model_path = os.path.join(model_path, MODEL_LEAD_NAME)

test_image_list = Image_Processing.extraction_image(test_path)
random.shuffle(test_image_list)
for i in test_image_list[:50]:
    Distinguish_image.distinguish_image(model_path, i)
model = tf.keras.models.load_model(model_path)
logger.info(model.evaluate(test_dataset))
