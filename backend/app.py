from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import os
import uuid

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('bilstm_urbansound8k_model100.h5')

class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                'siren', 'street_music']

def preprocess_audio(audio_file_path):
    temp_wav = 'temp.wav'
    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(22050).set_channels(1)

        target_duration_ms = 4000
        if len(audio) < target_duration_ms:
            audio += AudioSegment.silent(duration=target_duration_ms - len(audio))
        else:
            audio = audio[:target_duration_ms]

        audio.export(temp_wav, format='wav')

        signal, sr = librosa.load(temp_wav, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T

        target_time_steps = 173
        if mfcc.shape[0] > target_time_steps:
            mfcc = mfcc[:target_time_steps, :]
        elif mfcc.shape[0] < target_time_steps:
            mfcc = np.pad(mfcc, ((0, target_time_steps - mfcc.shape[0]), (0, 0)), mode='constant')

        input_data = np.expand_dims(mfcc, axis=0)

        os.remove(temp_wav)
        return input_data
    except Exception as e:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise Exception(f"Preprocessing error: {str(e)}")

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    file_ext = audio_file.filename.split('.')[-1].lower()

    try:
        # Use a unique filename to avoid conflicts
        temp_input_path = f'temp_input_{uuid.uuid4()}.{file_ext}'
        audio_file.save(temp_input_path)

        # Preprocess and predict
        input_data = preprocess_audio(temp_input_path)
        prediction = model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][class_idx])
        result = class_labels[class_idx]

        os.remove(temp_input_path)
        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
