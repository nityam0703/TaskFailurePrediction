import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('data2.csv')
data=data[:10000]
# Preprocess the data
X = data.drop(['Unnamed: 0','failed'], axis=1)
y = data['failed']

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape the data for LSTM input
X_train = np.reshape(np.array(X_train), (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(np.array(X_test), (X_test.shape[0], X_test.shape[1], 1))
# Define the Bi-LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.001))
model.add(Bidirectional(LSTM(60)))
model.add(Dropout(0.001))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=75, validation_data=(X_test, y_test))

# Evaluate the model
#loss, accuracy = model.evaluate(X_test, y_test)
#print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss, accuracy))
# Generate predictions for the test set
y_pred = model.predict(X_test)

# Convert predictions to binary values using a threshold of 0.5
y_pred = (y_pred > 0.5).astype(int)

# Compute the classification accuracy
a3 = accuracy_score(y_test, y_pred)

# Print the test loss and accuracy
loss = model.evaluate(X_test, y_test)
print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss, a3))


