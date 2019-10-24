from keras.models import Sequential
from keras import layers
# 这个脚本是用来训练用的
# 训练集就是一个文件夹，里面又有两个文件夹
# 每个文件夹各自放同一类图片

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

# ---------- ----------
train_dir = 'D:\\testPicture\\cat_dog\\trainData'
validation_dir = 'D:\\testPicture\\cat_dog\\validationData'


from keras_preprocessing.image import  ImageDataGenerator
train_dataGen = ImageDataGenerator(rescale=1./255)
test_dataGen = ImageDataGenerator(rescale=1./255)

train_generator = train_dataGen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_dataGen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break


history = model_cat_dog.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model_cat_dog.save('cats_and_dogs_sort.h5')