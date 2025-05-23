from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the model
model = tf.keras.models.load_model('lstm_urbansound8k_model100.h5')

# Class labels for UrbanSound8K
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                'siren', 'street_music']

# Preprocessing function
def preprocess_audio(audio_file_path):
    try:
        # Load audio with pydub to handle various formats
        audio = AudioSegment.from_file(audio_file_path)
        # Standardize to 22.05kHz, mono
        audio = audio.set_frame_rate(22050).set_channels(1)
        # Enforce 4-second duration
        target_duration_ms = 4000  # 4 seconds in milliseconds
        if len(audio) < target_duration_ms:
            audio = audio + AudioSegment.silent(duration=target_duration_ms - len(audio))
        else:
            audio = audio[:target_duration_ms]
        
        # Export to temporary WAV
        temp_wav = 'temp.wav'
        audio.export(temp_wav, format='wav')

        # Load with librosa
        signal, sr = librosa.load(temp_wav, sr=22050)
        
        # Extract MFCCs (13 coefficients, matching training)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        # mfcc shape: (13, frames)
        mfcc = mfcc.T  # Shape: (frames, 13)
        
        # Pad to max_len (use 173 or value from max_len script)
        target_time_steps = 173  # Update if max_len differs
        if mfcc.shape[0] > target_time_steps:
            mfcc = mfcc[:target_time_steps, :]  # Truncate
        elif mfcc.shape[0] < target_time_steps:
            mfcc = np.pad(mfcc, ((0, target_time_steps - mfcc.shape[0]), (0, 0)), mode='constant')
        
        # Reshape to (1, time_steps, n_mfcc) for LSTM
        input_data = np.expand_dims(mfcc, axis=0)  # Shape: (1, 173, 13)
        
        # Verify shape
        print("Preprocessed input shape:", input_data.shape)
        
        # Clean up temporary file
        os.remove(temp_wav)
        return input_data
    except Exception as e:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise Exception(f"Preprocessing error: {str(e)}")

@app.route('/classify', methods=['POST'])
def classify_audio():
    print('Hello')
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    # Validate file extension
    allowed_extensions = ['wav', 'mp3']
    if audio_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format. Use WAV or MP3'}), 400

    try:
        # Save uploaded file temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        # Preprocess and classify
        input_data = preprocess_audio(temp_path)
        prediction = model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][class_idx])
        result = class_labels[class_idx]
        print('HI')
        # Clean up
        os.remove(temp_path)
        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)