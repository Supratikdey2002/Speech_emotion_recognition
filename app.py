from flask import Flask, request, jsonify, render_template
from sklearn import *
import pickle
import numpy as np
import librosa
import os

# Load the trained Speech Emotion Recognition model
model_path = "modelForPrediction1.sav"
with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an audio file is part of the request
        if "audio" not in request.files:
            return render_template(
                "index.html", prediction_text="No audio file uploaded"
            )

        file = request.files["audio"]

        # Check if the file is empty
        if file.filename == "":
            return render_template(
                "index.html", prediction_text="No audio file selected"
            )

        # Save the audio file to a temporary location
        audio_path = os.path.join("temp", file.filename)
        file.save(audio_path)

        # Extract features from the audio file
        features = extract_features(audio_path)

        # Make prediction
        prediction = model.predict([features])
        emotion = get_emotion_label(prediction[0])

        # Clean up the temporary audio file
        os.remove(audio_path)

        return render_template("index.html", prediction_text=f"Prediction: {emotion}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


def extract_features(file_name):
    """
    Extract features from an audio file using librosa.
    """
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")

    # Extract MFCC features
    # Increase n_mfcc to 60 or as needed
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled = np.mean(chroma.T, axis=0)

    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_scaled = np.mean(mel.T, axis=0)

    # Combine all features into a single array
    features = np.hstack([mfccs_scaled, chroma_scaled, mel_scaled])

    # Ensure features are 180-dimensional by padding if necessary
    features = pad_features(features, 180)

    return features


def pad_features(features, target_size):
    """
    Pads or truncates the feature vector to a target size.
    """
    current_size = len(features)
    if current_size < target_size:
        # Pad with zeros if the current size is less than the target
        features = np.pad(features, (0, target_size - current_size), mode="constant")
    else:
        # Truncate if the current size is more than the target
        features = features[:target_size]
    return features


def get_emotion_label(label):
    """
    Convert numeric label to emotion string label.
    Modify this mapping based on your model's label encoding.
    """
    emotion_labels = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }
    label_str = f"{int((label)):10}"
    return emotion_labels.get(label_str, "Unknown")


if __name__ == "__main__":
    # Ensure the temp directory exists
    if not os.path.exists("temp"):
        os.makedirs("temp")

    app.run(debug=True)
