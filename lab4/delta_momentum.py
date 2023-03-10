import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

np.random.seed(42)

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
    # Define the input and output data
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))

    # Compile the model
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, verbose=1)
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    #Classification report
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5)
    target_names = ['benign', 'malignant']
    print(classification_report(y_val, y_pred, target_names=target_names))
    #Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:")
    print(cm)


    plot_training_curve(history)
    #print_model_weights(model)

if __name__ == "__main__":
    print(sys.argv[1:])
    train_perceptron()
