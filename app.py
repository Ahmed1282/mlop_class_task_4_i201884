from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data['text']
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    
    # Convert prediction to a JSON serializable type
    prediction_str = "positive" if prediction == 1 else "negative"  # Assuming 1 represents positive sentiment and 0 represents negative sentiment
    
    return jsonify({'sentiment': prediction_str})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
