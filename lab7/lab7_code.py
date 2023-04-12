import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, LayerNormalization
from keras.optimizers import SGD, Adam, Adamax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from keras.datasets import fashion_mnist

np.random.seed(42)
tf.random.set_seed(314)

def plot_training_curve(history):
    # Plot the training and validation loss
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train_loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True)
    ax.legend()
    
    # Plot the training and validation accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='train_acc')
    ax.plot(history.history['val_accuracy'], label='val_acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.grid(True)
    ax.legend()
    plt.show()

def train_perceptron():    
    # Load data and split into training and validation sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    #Format & normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    #Labels to one hot
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # Create model architecture
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Choose optimizer and compile the model
    learning_rate = 0.001
    sgd = SGD(lr=learning_rate,momentum=0.8)
    adam = Adam(lr=learning_rate)
    adamax = Adamax()
    model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
    
    #Check model summary!
    print(model.summary())
    
    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, verbose=1, callbacks=[early_stop], batch_size=4096)
    
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    #Classification report
    y_pred = model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_val = np.argmax(y_val, axis=1)
    print(y_val)
    print(y_pred)
    print(classification_report(y_val, y_pred))
    #Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:")
    print(cm)
    #Curves
    plot_training_curve(history)

if __name__ == "__main__":
    train_perceptron()
