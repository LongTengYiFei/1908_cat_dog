from keras import models
import tkinter as tk
from tkinter import filedialog
# 本脚本属于 单张测试， 如需多张测试 请另行编写脚本。*************************************
# 怎么使用这个脚本呢，启动它就行了，会弹出来一个文件选择器。

# 这里的模型路径，我们采用了硬编码模式。其实手动选择也可以。
model_cat_dog = models.load_model('D:\\111_PythonProjects\\1908\\cats_and_dogs_sort.h5')

import numpy as np
from PIL import Image

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

image_opened = Image.open(file_path)
image_resized = image_opened.resize((150, 150), Image.ANTIALIAS)
image_arrayed = np.array(image_resized)
print("Image resized and arrayed 's shape:", image_arrayed.shape)

# 就算我们想预测仅仅一张图片，我们也要把图片放进一个数组里面。
# 要注意放入的图片的大小，长宽，要符合神经网络的输入
img_list = np.array([image_arrayed, ])
print("The shape of input of net:", img_list.shape)
result = model_cat_dog.predict(img_list)
print("Shape of result:", result.shape)
print("The result is:", result)
# -------------------------------------------



print("* * * * * * * * * * * * * * * * * ")
if result[0][0] == 0:
    print("* * * * *  This is a cat! * * * *")
else:
    print("* * * * *  This is a dog! * * * *")
print("* * * * * * * * * * * * * * * * * ")
