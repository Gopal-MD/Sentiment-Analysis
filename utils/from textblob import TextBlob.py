from textblob import TextBlob

def extract_emotion_sentiment(text):
    """
    Extracts emotion and sentiment from a given text using TextBlob.
    Returns a tuple of (emotion, sentiment).
    """
    # Sentiment polarity: -1 (negative), 0 (neutral), +1 (positive)
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        sentiment_label = "positive"
    elif sentiment < 0:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Emotion categorization (simple example)
    if any(word in text.lower() for word in ["happy", "joy", "excited", "love"]):
        emotion = "happiness"
    elif any(word in text.lower() for word in ["sad", "cry", "upset", "heartbroken"]):
        emotion = "sadness"
    elif any(word in text.lower() for word in ["angry", "mad", "furious"]):
        emotion = "anger"
    elif any(word in text.lower() for word in ["fear", "scared", "terrified"]):
        emotion = "fear"
    else:
        emotion = "neutral"

    return emotion, sentiment_label