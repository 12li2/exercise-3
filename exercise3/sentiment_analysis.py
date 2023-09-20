
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Read Moby Dick file from Gutenberg dataset
file_path = nltk.data.find('D:/exercise3/nltk_data/corpora/gutenberg/melville-moby_dick.txt')
with open(file_path, 'r') as file:
    moby_dick_text = file.read()

# Tokenization
tokens = word_tokenize(moby_dick_text)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = [sia.polarity_scores(token)['compound'] for token in filtered_tokens]

# Calculate average sentiment score
average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

# Determine overall text sentiment
overall_sentiment = 'positive' if average_sentiment_score > 0.05 else 'negative'

# Display results
print("Average Sentiment Score:", average_sentiment_score)
print("Overall Text Sentiment:", overall_sentiment)
