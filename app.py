import os
import json
import operator
import tensorflow as tf
from flask import Flask
from flask import request
from loguru import logger
from models import Model
from settings import MODEL
from settings import MODEL_NAME
from settings import checkpoint_path
from settings import App_model_path
from Callback import CallBack
from Function_API import Distinguish_image

app = Flask(__name__)
if os.listdir(App_model_path):
    model_path = (os.path.join(App_model_path, os.listdir(App_model_path)[0]))
    logger.debug(f'{model_path}模型加载成功')
else:
    model = operator.methodcaller(MODEL)(Model)
    try:
        model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
    except:
        raise OSError(f'没有任何的权重和模型在{App_model_path}')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])
    model_path = os.path.join(App_model_path, MODEL_NAME)
    model.save(model_path)
    logger.debug(f'{model_path}模型加载成功')


@app.route("/", methods=['POST'])
def captcha_predict():
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False}
    get_data = request.form.to_dict()
    if 'img' in get_data.keys():
        base64_str = request.form['img']
        try:
            return_dict['result'] = Distinguish_image.distinguish_api(model_path=model_path, base64_str=base64_str)
        except Exception as e:
            return_dict['result'] = str(e)
            return_dict['return_info'] = '模型识别错误'
    else:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '参数错误，没有img属性'
    return json.dumps(return_dict, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
