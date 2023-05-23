import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer


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

def print_model_weights(model):
    weights = model.get_weights()
    print("Weights:", weights[0])
    print("Bias:", weights[1])

def train_perceptron():
    from sklearn.datasets import fetch_covtype
    dataset = fetch_covtype()
    X, y = dataset.data, dataset.target

    #vectorizer = CountVectorizer(max_df=0.3)
    #x_vec = vectorizer.fit_transform(X)
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #Baseline
    #svm = SVC(kernel='linear')
    #svm.fit(X_train,y_train)
    #y_pred = svm.predict(X_val)
    #print('Linear',classification_report(y_val, y_pred))
    #
    #svm = SVC(kernel='rbf')
    #svm.fit(X_train,y_train)
    #y_pred = svm.predict(X_val)
    #print('RBF',classification_report(y_val, y_pred))

    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_val)
    print('DT',classification_report(y_val, y_pred))

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # Create the model
    model = Sequential()
    #model.add(Dense(20, input_dim=X.shape[1], activation='softmax'))
    model.add(Dense(8, input_dim=X_train.shape[1], activation='softmax'))

    # Compile the model
    learning_rate = 0.0001
    sgd = SGD(lr=learning_rate)
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    
    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=1, callbacks=[early_stop], batch_size=4096)
    
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    #Classification report
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_val = np.argmax(y_val, axis=1)
    print(y_val)
    print(y_pred)
    print(classification_report(y_val, y_pred))
    #Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:")
    print(cm)

    plot_training_curve(history)
    #print_model_weights(model)

if __name__ == "__main__":
    train_perceptron()
