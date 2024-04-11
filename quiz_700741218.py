import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('weather.csv')

# Handle missing values
data.dropna(inplace=True)  # Drop rows with missing values

# Select relevant features and target variable
# Example: assuming 'temperature' is the target variable and other columns are features
features = data[['Data.Temperature.Avg Temp', 'Data.Temperature.Max Temp', 'Data.Temperature.Min Temp']]  # Select features
target = data['Data.Precipitation']  # Select target variable

# Scale numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Define sequence length (time steps) for LSTM model
seq_length = 10  # Example: use 10 time steps

# Reshape data into sequences
X = []
y = []
for i in range(len(scaled_features) - seq_length):
    X.append(scaled_features[i:i+seq_length])
    y.append(target[i+seq_length])

X = np.array(X)
y = np.array(y)

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Print shapes of training and test sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Define the LSTM-based architecture
model = Sequential()

# Add the first LSTM layer with dropout
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Add the second LSTM layer with dropout
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))

# Add the third LSTM layer with dropout
model.add(LSTM(units=64))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
print(model.summary())
