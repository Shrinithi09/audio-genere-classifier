#  Audio Genre Classifier (Rock vs Hip-Hop)

A machine learning web application that classifies uploaded audio files into **Rock** or **Hip-Hop** using extracted audio features and a trained Logistic Regression model.

🔗 Live App:https://audio-genere-classifier-z3yczobkvcn59ifgwxgk4b.streamlit.app/

---

## Project Overview

This project builds an end-to-end machine learning pipeline:

1. Audio feature extraction using librosa
2. Feature scaling and dimensionality reduction using StandardScaler + PCA
3. Model training using Logistic Regression
4. Model evaluation using 10-fold Cross Validation
5. Deployment as a live web application using Streamlit Cloud

---

## 🧠 Features Used

The model uses the following audio features:

- Tempo
- Energy (RMS)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Zero Crossing Rate
- MFCC 1
- MFCC 2

---

## 📊 Model Performance

- Algorithm: Logistic Regression
- Cross-validation: 10-fold K-Fold
- Mean Accuracy: ~79%

The final model was trained using a preprocessing pipeline and deployed for real-time prediction.

---

## Tech Stack

- Python
- scikit-learn
- librosa
- NumPy
- Streamlit
- Joblib

---

## Deployment

The application is deployed using Streamlit Cloud.

Users can:
1. Upload a .wav or .mp3 file
2. Automatically extract audio features
3. Receive genre prediction instantly

---

## Project Structure

app.py  
audio_genre_model.pkl  
requirements.txt  
README.md  

---

## Limitations

- Supports only 2 genres (Rock, Hip-Hop)
- Uses classical ML model (not deep learning)
- Feature extraction uses 2 MFCC coefficients
- Genre classification is subjective and dataset-dependent

---

##  Future Improvements

- Support more genres
- Use CNN on spectrogram images
- Extract more MFCC features (13+)
- Improve model accuracy
- Add probability confidence score

---

##  Author

Shrinithi Mahalakshmi  
Machine Learning Enthusiast 
