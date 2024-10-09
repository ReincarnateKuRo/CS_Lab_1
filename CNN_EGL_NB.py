from keras.datasets import mnist
from emnist import extract_training_samples, extract_test_samples
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt

# 資料前處理
def data_preprocessing():
    # 載入 MNIST 資料集
    (X_Train_mnist, y_Train_mnist), (X_Test_mnist, y_Test_mnist) = mnist.load_data()
    # 載入 EMNIST 的 "balanced" 子集
    X_Train_emnist, y_Train_emnist = extract_training_samples('balanced')
    X_Test_emnist, y_Test_emnist = extract_test_samples('balanced')
    # 合併 MNIST 和 EMNIST 訓練及測試資料
    X_Train = np.concatenate((X_Train_mnist, X_Train_emnist), axis=0)
    y_Train = np.concatenate((y_Train_mnist, y_Train_emnist), axis=0)
    X_Test = np.concatenate((X_Test_mnist, X_Test_emnist), axis=0)
    y_Test = np.concatenate((y_Test_mnist, y_Test_emnist), axis=0) 
    # 重塑圖像資料以符合 CNN 的輸入需求，並進行標準化
    X_Train = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32') / 255
    X_Test = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32') / 255
    # 進行標籤的 one-hot 編碼
    y_Train = to_categorical(y_Train, num_classes=62)
    y_Test = to_categorical(y_Test, num_classes=62)
    # # # # # # # # # # # # # # # # # # #    
    # 使用 ImageDataGenerator 進行數據增強
    #datagen = ImageDataGenerator(
    #    rotation_range=10,
    #    width_shift_range=0.1,
    #    height_shift_range=0.1,
    #    zoom_range=0.1,
    #    horizontal_flip=False
    #)
    #datagen.fit(X_Train)
    # # # # # # # # # # # # # # # # # # #
    # 數據集資料前處理function終
    return X_Train, y_Train, X_Test, y_Test

# 架設CNN模型
def build_cnn_model():

    model = Sequential()
    # 卷積層1
    model.add(Conv2D(filters=32, 
                     kernel_size=(5,5), 
                     padding='same', 
                     input_shape=(28,28,1), 
                     activation='relu', 
                     name='conv2d_1'))
    # 池化層1
    model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))
    # 卷積層2
    model.add(Conv2D(filters=64, 
                     kernel_size=(5,5), 
                     padding='same', 
                     activation='relu', 
                     name='conv2d_2'))
    # 池化層2
    model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))
    # 卷積層3
    model.add(Conv2D(filters=128, 
                     kernel_size=(3,3), 
                     padding='same', 
                     activation='relu', 
                     name='conv2d_3'))
    # 池化層3
    model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_3'))
    # Dropout1
    model.add(Dropout(0.25, name='dropout_1'))
    # 平坦層
    model.add(Flatten(name='flatten_1'))
    # 隱藏層
    model.add(Dense(784, activation='relu', name='dense_1'))
    # Dropout2
    model.add(Dropout(0.4, name='dropout_2'))
    # 輸出層
    model.add(Dense(62, activation='softmax', name='dense_2'))
    # 輸出模型參數
    model.summary()
    # 定義訓練方式
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    # 模型構建function輸出
    return model

# 訓練與評估模型
def train_and_evaluate_model(model, X_Train, y_Train, X_Test, y_Test):
    # 開始訓練
    train_history = model.fit(
        x=X_Train, y=y_Train, batch_size=300,
        validation_split=0.2,
        epochs=2,
        verbose=1
    )

    # 評估模型
    scores = model.evaluate(X_Test, y_Test)
    print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
    # 訓練與測試結果Function終 
    return train_history, scores

# 繪製圖表
class Plotter:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    def __init__(self):
        pass  # 初始化可以保留空的，如果有需要也可以傳入參數

    def plot_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap='binary')
        plt.show()

    def plot_images_labels_predict(self, images, labels, prediction, idx, num=10):
        fig = plt.gcf()
        fig.set_size_inches(12, 14)
        if num > 25:
            num = 25
        for i in range(0, num):
            ax = plt.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = "l=" + str(labels[idx])
            if len(prediction) > 0:
                title = "l={}, p={}".format(str(labels[idx]), str(prediction[idx]))
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()

    def show_train_history(self, train_history, train, validation):
        epochs = len(train_history.history[train])
        x_ticks = np.arange(1, epochs + 1, 1)

        plt.plot(x_ticks, train_history.history[train])
        plt.plot(x_ticks, train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.xticks(x_ticks)

        y_min = min(min(train_history.history[train]), min(train_history.history[validation]))
        y_max = max(max(train_history.history[train]), max(train_history.history[validation]))
        plt.ylim(y_min, y_max)

        plt.show()

# 主程式 belike
if __name__ == "__main__":
    # 資料前處理
    X_Train, y_Train, X_Test, y_Test = data_preprocessing()

    # 架設 CNN 模型
    model = build_cnn_model()

    # 訓練與評估模型
    train_history, scores = train_and_evaluate_model(model, X_Train, y_Train, X_Test, y_Test)
    
    # 繪製圖表
    plotter = Plotter()
    plotter.show_train_history(train_history, 'accuracy', 'val_accuracy')
    plotter.show_train_history(train_history, 'loss', 'val_loss')
    
    # 儲存model
    model.save('hwr_model.h5')
    model.save_weights('hwr_model.weights.h5')




