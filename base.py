from flask import Flask, request, jsonify
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)

# Load the model and processor once during initialization
def load_model():
    global processor, model, label_mapping
    processor = Wav2Vec2Processor.from_pretrained("Non-playing-Character/emotion-speech")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("Non-playing-Character/emotion-speech")
    model.eval()
    label_mapping = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    print("Model loaded successfully")

def preprocess_audio(file_path):
    # Load audio file
    audio, _ = librosa.load(file_path, sr=16000)
    max_length = 160000

    # Process the audio file with padding and truncation
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Reshape input_values to be 2D: [batch_size, sequence_length]
    input_values = inputs["input_values"].squeeze(0)
    input_values = input_values.unsqueeze(0)

    return input_values

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # Save the file temporarily
    file_path = "./temp_audio.wav"
    file.save(file_path)

    # Preprocess the audio
    input_values = preprocess_audio(file_path)

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()
    prediction = label_mapping[predicted_label]

    # Return the result as JSON
    return jsonify({
        "predicted_label": prediction,
    })

if __name__ == '__main__':
    load_model()  # Load the model during initialization
    app.run(debug=True)
