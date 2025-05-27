from pydub import AudioSegment
import librosa
import numpy as np
import tensorflow as tf
import os
import sys

# Paths to the models
MODEL_PATHS = {
    'LSTM': 'D:/PROJECT11/backend/bilstm_urbansound8k_model.h5',
    'BiLSTM': 'D:/PROJECT11/backend/bilstm_urbansound8k_model100.h5',
    'CNN': 'D:/PROJECT11/backend/bilstm20.h5'
}

# Class labels
class_labels = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

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
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512).T

        target_time_steps = 173
        if mfcc.shape[0] < target_time_steps:
            mfcc = np.pad(mfcc, ((0, target_time_steps - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:target_time_steps]

        return np.expand_dims(mfcc, axis=0)

    except Exception as e:
        print(f"❌ Error in preprocessing: {e}", file=sys.stderr)
        return None

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def predict_with_model(model_path, model_name, input_data):
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(input_data, verbose=0)
    index = np.argmax(prediction)
    label = class_labels[index]
    confidence = float(prediction[0][index]) * 100
    return model_name, label, confidence

# Entry point
if __name__ == '__main__':
    audio_path = 'D:/PROJECT11/audio/fold2/203929-7-9-12.wav'
    input_data = preprocess_audio(audio_path)

    if input_data is not None:
        print(f"\n🎧 Results for audio: {audio_path}\n")
        for name, path in MODEL_PATHS.items():
            try:
                model_name, label, confidence = predict_with_model(path, name, input_data)
                print(f"🔹 {model_name} Prediction: {label} ({confidence:.2f}%)")
            except Exception as e:
                print(f"❌ Failed with {name}: {e}")
    else:
        print("⚠️ Could not generate prediction due to preprocessing error.")
