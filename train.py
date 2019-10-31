from keras.models import Sequential
from keras import layers
# 这个脚本是用来训练用的
# 训练集就是一个文件夹，里面又有两个文件夹
# 每个文件夹各自放同一类图片
import  tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

model_cat_dog = Sequential()
model_cat_dog.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model_cat_dog.add(layers.MaxPool2D(2, 2))
model_cat_dog.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cat_dog.add(layers.MaxPool2D(2, 2))
model_cat_dog.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cat_dog.add(layers.MaxPool2D(2, 2))
model_cat_dog.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cat_dog.add(layers.MaxPool2D(2, 2))

model_cat_dog.add(layers.Flatten())
model_cat_dog.add(layers.Dense(512, activation='relu'))
model_cat_dog.add(layers.Dense(1, activation='sigmoid'))

model_cat_dog.summary()

# ----------
from keras import  optimizers
model_cat_dog.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
import os
# 我能不能预先载入模型，继续训练？ 10月14日 17点57分
if os.path.exists('.\\cats_and_dogs_sort.h5') == True:
   print('模型文件已存在。')
   model_cat_dog.load_weights('.\\cats_and_dogs_sort.h5')
   print('模型参数载入成功')

# ---------- ----------
train_dir = 'D:\\testPicture\\cat_dog\\trainData'
validation_dir = 'D:\\testPicture\\cat_dog\\validationData'


from keras_preprocessing.image import  ImageDataGenerator
train_dataGen = ImageDataGenerator(rescale=1./255)
test_dataGen = ImageDataGenerator(rescale=1./255)

batch_size = int(input('batch的大小是多少？输入数字按回车结束：'))

train_generator = train_dataGen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

validation_generator = test_dataGen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break
steps_per_epoch = int(input('一个epoch走几个batch？输入数字按回车结束输入：'))
epochs = int(input('走几个epoch？输入数字按回车结束输入：'))
history = model_cat_dog.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=50
)

model_cat_dog.save('cats_and_dogs_sort.h5')