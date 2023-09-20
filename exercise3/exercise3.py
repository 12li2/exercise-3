import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from nltk.corpus import wordnet

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Read Moby Dick file from Gutenberg dataset
file_path = nltk.data.find('D:/exercise3/nltk_data/corpora/gutenberg/melville-moby_dick.txt')
with open(file_path, 'r') as file:
    moby_dick_text = file.read()

tokens = word_tokenize(moby_dick_text)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_counts.most_common(5)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = []
for word, pos in pos_tags[:20]:
    pos = pos[0].lower() if pos[0].lower() in ['a', 'r', 'n', 'v'] else 'n'
    lemma = lemmatizer.lemmatize(word, pos=pos)
    lemmatized_tokens.append(lemma)

# Plotting frequency distribution
pos_labels, pos_values = zip(*pos_counts.items())

plt.figure(figsize=(10, 5))
plt.bar(pos_labels, pos_values)
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of POS')
plt.xticks(rotation=45)
plt.show()

# Display results
print("Top 5 Parts of Speech and their Frequencies:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print("\nTop 20 Lemmatized Tokens:")
print(lemmatized_tokens)
