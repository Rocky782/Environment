import pandas as pd
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

# Load UrbanSound8K metadata
metadata = pd.read_csv('metadata/UrbanSound8K.csv')

# Extract MFCC features
def extract_features(file_name, n_mfcc=13, hop_length=512, n_fft=2048):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs = mfccs.T
        return mfccs
    except Exception as e:
        print("Error parsing file:", file_name)
        return None

# Process each audio file
features = []
labels = []
for index, row in metadata.iterrows():
    file_name = os.path.join('audio', str(row['fold']), str(row['slice_file_name']))
    data = extract_features(file_name)
    if data is not None:
        features.append(data)
        labels.append(row['classID'])

# Convert to numpy arrays and pad sequences
X = np.array(features)
y = np.array(labels)
max_len = max([x.shape[0] for x in X])
X_padded = [np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') for x in X]
X = np.array(X_padded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(128, input_shape=(max_len, 13)))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(64, activation='relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(10, activation='softmax'))  # 10 classes
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# BiLSTM Model
model_bilstm = Sequential()
model_bilstm.add(Bidirectional(LSTM(128), input_shape=(max_len, 13)))
model_bilstm.add(Dropout(0.3))
model_bilstm.add(Dense(64, activation='relu'))
model_bilstm.add(Dropout(0.3))
model_bilstm.add(Dense(10, activation='softmax'))
model_bilstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_bilstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))