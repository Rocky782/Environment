from pydub import AudioSegment  # ✅ Must be first before using AudioSegment
import librosa
import numpy as np
import tensorflow as tf
import os
import sys  # Optional: for better error output

# Load model
model = tf.keras.models.load_model('D:/PROJECT11/backend/lstm_urbansound8k_model100.h5')

# Class labels
class_labels = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

def preprocess_audio(audio_file_path):
    temp_wav = 'temp.wav'
    try:
        # Load and process audio
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(22050).set_channels(1)

        target_duration_ms = 4000  # 4 seconds
        if len(audio) < target_duration_ms:
            silence = AudioSegment.silent(duration=target_duration_ms - len(audio))
            audio += silence
        else:
            audio = audio[:target_duration_ms]

        audio.export(temp_wav, format='wav')

        # Load waveform
        signal, sr = librosa.load(temp_wav, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512).T

        # Pad or truncate
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

# Run prediction
if __name__ == '__main__':
    audio_path = 'D:/PROJECT11/audio/fold1/7061-6-0-0.wav'
    input_data = preprocess_audio(audio_path)

    if input_data is not None:
        prediction = model.predict(input_data, verbose=0)
        predicted_label = class_labels[np.argmax(prediction)]
        print(f"✅ Prediction: {predicted_label}")
    else:
        print("⚠️ Could not generate prediction due to preprocessing error.")
