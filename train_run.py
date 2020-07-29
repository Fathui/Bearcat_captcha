import os
import operator
import tensorflow as tf
from loguru import logger
from models import Model
from Callback import CallBack
from settings import MODEL
from settings import EPOCHS
from settings import BATCH_SIZE
from settings import model_path
from settings import MODEL_NAME
from settings import DATA_ENHANCEMENT
from settings import train_pack_path
from settings import validation_pack_path
from settings import test_pack_path
from settings import train_enhance_path
from Function_API import cheak_path
from Function_API import Image_Processing
from Function_API import parse_function_verification

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

train_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(train_pack_path)).map(
    parse_function_verification).batch(BATCH_SIZE)

validation_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(validation_pack_path)).map(
    parse_function_verification).batch(
    BATCH_SIZE)

test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
    parse_function_verification).batch(BATCH_SIZE)

model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Model))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['categorical_accuracy'])
if DATA_ENHANCEMENT:
    logger.debug(f'一共有{len(Image_Processing.extraction_image(train_dataset))}个batch')
else:
    logger.debug(f'一共有{len(Image_Processing.extraction_image(train_enhance_path))}个batch')

model.fit(train_dataset, epochs=EPOCHS, callbacks=c_callback, validation_data=validation_dataset, verbose=2)

save_model_path = cheak_path(os.path.join(model_path, MODEL_NAME))

model.save(save_model_path)

logger.info(model.evaluate(test_dataset))
