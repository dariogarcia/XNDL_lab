import sys
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curve(history):
    #Generate the accuracy/loss plot during training
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.title('Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_decision_boundary(model,X,y):
    w = model.get_weights()[0]
    b = model.get_weights()[1]
    x1 = np.linspace(0, 1, 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, 'k-')
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def print_model_weights(model):
    weights = model.get_weights()
    print("Weights:", weights[0])
    print("Bias:", weights[1])

def train_perceptron():
    # Define the input and output data
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    # Create the model
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))

    #print weights

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X, y, epochs=500, verbose=0)
    plot_training_curve(history)
    plot_decision_boundary(model,X,y)

    # Test the model
    predictions = model.predict(X)
    print(predictions)

    print_model_weights(model)

if __name__ == "__main__":
    print(sys.argv[1:])
    train_perceptron()
