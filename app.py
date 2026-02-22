import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)

    # Tempo (force scalar)
    tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])

    # Energy
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # MFCCs (first 2)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
    mfcc_1 = float(np.mean(mfcc[0]))
    mfcc_2 = float(np.mean(mfcc[1]))

    return np.array([
        tempo,
        energy,
        spectral_centroid,
        spectral_bandwidth,
        rolloff,
        zero_crossing_rate,
        mfcc_1,
        mfcc_2
    ], dtype=float)

#ui

import streamlit as st
import joblib
import numpy as np
import tempfile

model = joblib.load("audio_genre_model.pkl")

st.title("🎵 Audio Genre Classifier (Real-World Version)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    features = extract_features(audio_path)
    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    genre = "Rock" if prediction == 0 else "Hip-Hop"

    st.success(f"Predicted Genre: {genre}")
