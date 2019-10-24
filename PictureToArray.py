import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
"脚本介绍********************************"
"这个脚本可以把图片变成数组, 然后打印一点信息" \

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
Image_opened = Image.open(file_path)
arrayOfImage = np.array(Image_opened)

print("Type   OfImage:")
print(type(Image_opened))
print("--------------------")
print("")

print("Type   OfArrayOfImage  also the whole array:")
print(type(arrayOfImage))
print("--------------------")
print("")

print("Information   OfImage:")
print(Image_opened)
print("--------------------")
print("")

print("arrayOfImage:")
print(arrayOfImage)
print("--------------------")
print("")

print("arrayOfImage[0]:")
print(arrayOfImage[0])
print("--------------------")
print("")

print("Array   OfImage[0][0]  also  the first pixel:")
print(arrayOfImage[0][0])
print("--------------------")
print("")

print("Type   OfArrayOfImage[0][0] also the first pixel:")
print(type(arrayOfImage[0][0]))
print("As you know,it is a N dimension array.")
print("--------------------")
print("")

print("value Of    Image[0][0][0]:")
print(arrayOfImage[0][0][0])
print("--------------------")
print("")

print("value Of    Image[0][0][1]:")
print(arrayOfImage[0][0][1])
print("--------------------")
print("")

print("value Of    Image[0][0][2]:")
print(arrayOfImage[0][0][2])
print("--------------------")
print("")