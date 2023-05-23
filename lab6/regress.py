import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.initializers import he_normal, glorot_normal
from sklearn.linear_model import LinearRegression

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
    plt.show()

def print_model_weights(model):
    weights = model.get_weights()
    print("Weights:", weights[0])
    print("Bias:", weights[1])

def regression_report(model, X_train, y_train, X_test, y_test):
    # Predict on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print report
    print('Training MSE: {:.4f}'.format(train_mse))
    print('Testing MSE: {:.4f}'.format(test_mse))
    print('Training RMSE: {:.4f}'.format(train_rmse))
    print('Testing RMSE: {:.4f}'.format(test_rmse))
    print('Training MAE: {:.4f}'.format(train_mae))
    print('Testing MAE: {:.4f}'.format(test_mae))
    print('Training R-squared: {:.4f}'.format(train_r2))
    print('Testing R-squared: {:.4f}'.format(test_r2))

def train_perceptron():
    from sklearn.datasets import load_diabetes
    loaded_data = load_diabetes()
    #from sklearn.datasets import fetch_california_housing
    #loaded_data = fetch_california_housing()
    X, y = loaded_data.data, loaded_data.target
    print(X.shape)
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate input data
    theta = np.random.uniform(0, 2*np.pi, size=(1000, 1))
    r = np.sqrt(np.random.uniform(0, 1, size=(1000, 1)))
    X = np.concatenate((r*np.cos(theta), r*np.sin(theta)), axis=1)
    y = np.zeros((1000, 1))
    y[np.where(r <= 0.5)] = 1
    y += np.random.normal(0, 0.1, size=y.shape)

    # Split into training and testing sets
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    
    # Create the model
    model = Sequential()
    #model.add(Dense(32, input_dim=X.shape[1], activation='tanh'))
    #model.add(Dense(32, input_dim=X.shape[1], activation='tanh'))
    #model.add(Dense(4, activation='tanh',input_dim=X.shape[1]))
    model.add(Dense(4, activation='relu',input_dim=X.shape[1], kernel_initializer=he_normal()))
    model.add(Dense(1))
    #model.add(Dense(4, activation='tanh'))
    #model.add(Dense(1, activation='tanh'))

    # Compile the model
    learning_rate = 0.01
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum)
    adam = Adam(lr = learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=1, callbacks=[early_stop], batch_size=64)
    # Evaluate the model on the validation set
    loss, mse = model.evaluate(X_val, y_val, verbose=1)
    print("Validation loss: {:.3f}, Validation mse: {:.3f}".format(loss, mse))
    regression_report(model,X_train,y_train,X_val,y_val)
    #Classification report
    #y_pred = model.predict(X_val)
    #y_pred = (y_pred > 0.5)
    #target_names = ['benign', 'malignant']
    #print(reclassification_report(y_val, y_pred, target_names=target_names))
    #Confusion matrix
    #cm = confusion_matrix(y_val, y_pred)
    #print("Confusion matrix:")
    #print(cm)

    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_val)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_val, y_pred)
    print("Baseline Mean Squared Error: ", mse)


    plot_training_curve(history)
    #print_model_weights(model)

if __name__ == "__main__":
    print(sys.argv[1:])
    train_perceptron()
