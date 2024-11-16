from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Sentiment analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    tweet = data.get('tweet', '')
    if not tweet:
        return jsonify({'sentiment': 'No input provided'})

    # Use the pre-trained model for sentiment analysis
    result = sentiment_analyzer(tweet)

    # Extract sentiment and confidence score from the result
    sentiment = result[0]['label']
    confidence = result[0]['score']

    return jsonify({'sentiment': sentiment, 'confidence': confidence})

if __name__ == "__main__":
    app.run(debug=True)
