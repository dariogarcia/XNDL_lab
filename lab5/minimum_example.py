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

np.random.seed(42)
tf.random.set_seed(314)

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

    # Create the model
    model = Sequential()
    model.add(Dense(2000, activation='relu',input_dim=X.shape[1], kernel_initializer=he_normal()))

    # Compile the model
    learning_rate = 0.1
    momentum = 0.89
    adam = Adam(lr = learning_rate)
    sgd = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])

    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=1, callbacks=[early_stop], batch_size=64)
    # Evaluate the model on the validation set
    loss, mse = model.evaluate(X_val, y_val, verbose=1)
    print("Validation loss: {:.3f}, Validation mse: {:.3f}".format(loss, mse))
    regression_report(model,X_train,y_train,X_val,y_val)

if __name__ == "__main__":
    print(sys.argv[1:])
    train_perceptron()
