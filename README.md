# HairEditing
根目录: ```HairEditing``` 

使用方式：
  1. 将人脸图像和头发图像分别复制到 ```./data/face```和```./data/hair``` 下
  2. 运行代码：```python3 ./src/main.py 人脸图片 头发图片```
 
 样例：
  ```python3 ./src/main.py 00013.png 00066.png```
  其中 00013.png 在```./data/face``` 下
  00066.png 在```./data/hair``` 下

 输出：
 输出位置为：```./results```中
 1. 文件夹名称为：```人脸图片名称_头发图片名称```
 2. 文件夹内最终结果为```final.png```

上述例子中文件在```./results/00013_00066/```文件夹中

当前仅实现了头发替换，没有头发编辑的功能。同时提供的人脸图片和头发图片需要满足ffhq的人脸对齐标准

pretrain model的链接：https://drive.google.com/file/d/1VpcT5fqqypHWj3znABBtIl9yAgzE_FIF/view?usp=sharing

将解压后的```pretrained_model```文件夹放到根目录中
