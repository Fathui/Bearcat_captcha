'''
打包数据
'''
import random
from loguru import logger
from settings import train_path
from settings import validation_path
from settings import test_path
from settings import IMAGE_HEIGHT
from settings import IMAGE_WIDTH
from settings import train_enhance_path
from settings import DATA_ENHANCEMENT
from settings import TFRecord_train_path
from settings import TFRecord_validation_path
from settings import TFRecord_test_path
from Function_API import Image_Processing
from Function_API import WriteTFRecord
from concurrent.futures import ThreadPoolExecutor

if DATA_ENHANCEMENT:
    with ThreadPoolExecutor(max_workers=100) as t:
        for i in Image_Processing.extraction_image(train_path):
            task = t.submit(Image_Processing.preprosess_save_images, i, [IMAGE_HEIGHT, IMAGE_WIDTH])
    train_image = Image_Processing.extraction_image(train_enhance_path)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_one_hot_lable(train_image)
else:
    train_image = Image_Processing.extraction_image(train_path)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_one_hot_lable(train_image)

validation_image = Image_Processing.extraction_image(validation_path)
validation_lable = Image_Processing.extraction_one_hot_lable(validation_image)

test_image = Image_Processing.extraction_image(test_path)
test_lable = Image_Processing.extraction_one_hot_lable(test_image)
logger.debug(train_image)
# logger.debug(train_lable)

with ThreadPoolExecutor(max_workers=3) as t:
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_train_path, train_image, train_lable)
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_validation_path, validation_image, validation_lable)
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_test_path, test_image, test_lable)
