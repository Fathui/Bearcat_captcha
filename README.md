#熊猫不定长验证码识别

#项目基于tensorflow2.1
    项目的输入图片的格式为.jpg
    不是.jpg后缀也不用慌本项目有修改后缀的代码
    后面会介绍

##项目启动
    ps:不想自己练的直接运行app.py
    默认开启5006端口,post请求接受一个参数img
    需要base64一下,具体请看spider_example.py
    
##第一步:到项目路径安装项目运行所需环境
###最好用虚拟环境
    
    CPU训练
    pip install -r requirements.txt -i https://pypi.douban.com/simple/

    GPU训练
    1.安装Anconda
    2.安装tensorflow2.1
    conda install tensorflow
    pip install -r requirements.txt -i https://pypi.douban.com/simple/
    
##第二步:初始化工作路径

    运行init_working_space.py
    python init_working_space.py

##第三步:准备标注好的数据(注意数据不能放太深,一个文件夹下面就放上数据)

    1.将训练数据放到train_dataset文件夹

    2.将验证数据放到validation_dataset文件夹

    3.将测试数据放到test_dataset文件夹

##如果你的标注数据是一坨的话按照下面步骤区分开来

    1.将一坨数据放到train_dataset文件夹

    2.运行move_path.py
      python move_path.py

##如果你暂时没有数据,不用慌,先用生成的数据集吧

    运行gen_sample_by_captcha.py

##第四步:修改配置文件

    这个后面在详细说先用默认设置启动项目吧

##第五步:打包数据

    运行pack_dataset.py
    python pack_dataset.py

##第六步:添加模型(model)

    暂时先使用项目自带的模型吧

##第七步:编译模型

    暂时先默认吧

##第八步:开始训练

    运行train_run.py
    python train_run.py
    
##第九步:开启可视化

tensorboard --logdir "logs"

##第十步:评估模型

    丹药出来后要看一下是几品丹药
    运行read_model.py

##第十一步:开启后端

    运行app.py
    python app.py
    
##第十二步:调用接口

    先运行本项目给的例子感受一下
    python spider_example.py

##下面开始补充刚刚省略的一些地方,由于设置文件备注比较完善，解释部分参数

### 是否使用数据增强(数据集多的时候不需要用)
    DATA_ENHANCEMENT = False
数据集不够或者过拟合时，可以考虑数据增强下

增强方法在Function_API.py里面的Image_Processing.preprosess_save_images

### 验证码的长度
    CAPTCHA_LENGTH = 6
    
这个数字要取你要识别验证码的最大长度,不足的会自动用'_'补齐

否则会报错raise ValueError

### BATCH_SIZE

    BATCH_SIZE = 16

如果你的显卡很牛逼，可以尝试调大点

### 训练次数

    EPOCHS = 100

请放心调有

训练多少轮验证损失下不去，停止训练的回调设置

    EARLY_PATIENCE = 8
    
定义模型的方法名字,模型在models.py里的Model类

    MODEL = 'captcha_model'

其他设置如果没有特别情况，尽量不要改

##接下来说明项目的文件夹，及文件

##文件夹

###App_model
    后端模型保存路径

###checkpoint
    保存检查点
    
###CSVLogger
    把训练轮结果数据流到 csv 文件

###logs
    保存被 TensorBoard 分析的日志文件

###model
    保存模型
    
###train_dataset
    保存训练集
    
###train_enhance_dataset
    保存增强后的训练集
    
###train_pack_dataset
    保存打包好的训练集
    
###validation_dataset
    保存验证集
    
###vailidation_pack_dataset
    保存打包好的验证集
    
###test_dataset
    保存测试集
    
###test_pack_dataset
    保存打包好的测试集
    
##文件

###app.py
    开启后端

###Callback.py
    回调函数参考
    [keras中文官网](https://keras.io/zh/callbacks/)
    运行该文件会返回一个最佳的权重文件
    
###captcha_config.json
    生成验证码的配置文件
      "image_suffix": "jpg",生成验证码的后缀
      "count": 20000,生成验证码的数量
      "char_count": 4,生成验证码的长度
      "width": 100,生成验证码的宽度
      "height": 40，生成验证码的高度

###del_file.py
    删除所有数据集的文件
    这里是防止数据太多手动删不动
    
###Function_API.py
    项目核心，三大类
    Image_Processing
    图片处理和标签处理
    WriteTFRecord
    打包数据集
    Distinguish_image
    预测类模型生成后用这个类来预测和部署
    
###gen_sample_by_captcha.py
    生成验证码
    
###init_working_space.py
    初始化工作目录
    ***注意:此文件只在第一次运行项目时运行***
    ***因为这会重置checkpoint CSVLogger logs***
    
###models.py
    搭建模型网络
    
###pack_dataset.py
    打包数据集
  
###rename_suffix.py
    修改训练集文件为.jpg后缀
    验证集文件和测试集文件有需要修改后缀自行改代码
    
###settings.py
    项目的设置文件
    
###spider_example.py
    爬虫调用例子
    
###sub_filename.py
    替换文件名
    例如文件名为test.01.jpg
    运行后会修改为
    test_01.jpg

###test_model.py
    读取模型进行测试

###train_run.py
    开始训练

特别感谢下面一些项目对我的启发

[安师大教务系统验证码检测](https://github.com/AHNU2019/AHNU_captcha)

[cnn_captcha](https://github.com/nickliqian/cnn_captcha)

[captcha_trainer](https://github.com/kerlomz/captcha_trainer)

[captcha-weibo](https://github.com/skygongque/captcha-weibo/blob/master/client.py)

感谢大佬们的数据集让我省去很多成本和时间

搜狗验证码链接：https://pan.baidu.com/s/13wMK3GXaTZ-yaX0vNDG7Ww 提取码：9uxv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-149.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

微博验证码链接：https://pan.baidu.com/s/1w5-MMzX47US3GS8a7xlSBw 提取码: 74uv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-470.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

###***此项目以研究学习为目的，禁止用于非法用途***
###再次说明项目的tensorflow的版本是2.1不要搞错了
2.2也可以

###经过我的训练，微博加搜狗的验证码正确率达到了97.5%

###模型保存在App_model文件夹里,与大家共同学习

###ps:新手上路，轻喷
