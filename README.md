# HairEditing
根目录: ```HairEditing``` 

使用方式：
  1. 将人脸图像和头发图像分别复制到 ```./data/face```和```./data/hair``` 下
  2. 运行代码：```python3 ./src/main.py 人脸图片 头发图片```
 
 样例：
  ```python3 ./src/main.py 00013.png 00020.png```
  其中 00013.png 在```./data/face``` 下
  00020.png 在```./data/hair``` 下

 输出：
 输出位置为：```./results```中
 1. 文件夹名称为：```人脸图片名称_头发图片名称```
 2. 文件夹内最终结果为```final.png```

上述例子中文件在```./results/00013_00020/```文件夹中