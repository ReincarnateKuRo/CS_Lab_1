from keras.datasets import mnist
from emnist import extract_training_samples, extract_test_samples
import numpy as np

# 載入 MNIST 資料集
(X_Train_mnist, y_Train_mnist), (X_Test_mnist, y_Test_mnist) = mnist.load_data()

# 載入 EMNIST 的 "balanced" 子集，這個子集包含數字和大小寫英文字母
X_Train_emnist, y_Train_emnist = extract_training_samples('balanced')
X_Test_emnist, y_Test_emnist = extract_test_samples('balanced')

# 將 MNIST 和 EMNIST 的訓練資料及標籤合併
X_Train = np.concatenate((X_Train_mnist, X_Train_emnist), axis=0)
y_Train = np.concatenate((y_Train_mnist, y_Train_emnist), axis=0)

X_Test = np.concatenate((X_Test_mnist, X_Test_emnist), axis=0)
y_Test = np.concatenate((y_Test_mnist, y_Test_emnist), axis=0)

X_Train = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32') / 255
X_Test = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32') / 255

#參考
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,  # 隨機旋轉範圍
    width_shift_range=0.1,  # 隨機水平移動
    height_shift_range=0.1,  # 隨機垂直移動
    zoom_range=0.1,  # 隨機縮放
    horizontal_flip=False  # 不隨機翻轉
)

datagen.fit(X_Train)



from keras.utils import to_categorical

y_Train = to_categorical(y_Train, num_classes=62)
y_Test = to_categorical(y_Test, num_classes=62)



from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization

model = Sequential()
## 卷積層1(預期要增加filter)
model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu',
                 name='conv2d_1'))



## 池化層1
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))

## 卷積層2
model.add(Conv2D(filters=64,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu',
                 name='conv2d_2'))



## 池化層2
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))

## 卷積層3
model.add(Conv2D(filters=128, 
                 kernel_size=(3, 3), 
                 padding='same', 
                 input_shape=(28,28,1),
                 activation='relu',
                 name='conv2d_3'))
## 池化層3
model.add(MaxPool2D(pool_size=(2, 2),name='max_pooling2d_3'))

## Dropout1
model.add(Dropout(0.25, name='dropout_1'))

## 平坦層
model.add(Flatten(name='flatten_1'))

## 隱藏層
model.add(Dense(784, activation='relu', name='dense_1'))

## dropout2
model.add(Dropout(0.4, name='dropout_2'))


## 輸出層26+26+10=62
model.add(Dense(62, activation='softmax', name='dense_2'))  # 62 類別

#輸出模型參數
model.summary()
print("")

#定義訓練方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 開始訓練
train_history = model.fit(x=X_Train,
                          y=y_Train, validation_split=0.2,
                          epochs=5, batch_size=300, verbose=1)

#評估模型
scores = model.evaluate(X_Test, y_Test)
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))







