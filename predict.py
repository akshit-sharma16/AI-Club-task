import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = '/content/drive/MyDrive/SER_Project_Model.keras' #choose weights file path here
EMOTIONS = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
            4: 'angry', 5: 'fear', 6: 'disgust', 7: 'surprise'}

def predict_emotion(wav_file_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    audio, sr = librosa.load(wav_file_path, res_type='soxr_hq')
    audio, _ = librosa.effects.trim(audio)

    fixed_length = 22050 * 3
    if len(audio) > fixed_length:
        audio = audio[:fixed_length]
    else:
        audio = np.pad(audio, (0, fixed_length - len(audio)), 'constant')

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    features = log_mel[np.newaxis, ..., np.newaxis]

    predictions = model.predict(features, verbose=0)
    score = np.max(predictions)
    emotion_id = np.argmax(predictions)

    predicted_emotion = EMOTIONS[emotion_id]

    print(f"--- 🎤 Inference Result ---")
    print(f"File: {wav_file_path.split('/')[-1]}")
    print(f"Predicted Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {score * 100:.2f}%")
    print("---------------------------")

# to test on a file do
# predict_emotion('file_path')
