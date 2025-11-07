import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure 'stopwords' are downloaded
nltk.download('stopwords', quiet=True)

# Load English stop words
stop_words = set(stopwords.words('english'))

# Read input file
with open(r"C:\Users\amrut\OneDrive\Desktop\IR\sample.txt", "r", encoding="utf-8") as file1:
    text = file1.read()

# Tokenize words
words = word_tokenize(text)

print(f"\nBefore Removing Stop Words:\n{words}\nLength: {len(words)}")

# Filter out stop words
filtered_words = [word for word in words if word.lower() not in stop_words]

# Write filtered words to output file
with open(r"C:\Users\amrut\OneDrive\Desktop\IR\sample.txt", "w", encoding="utf-8") as file2:
    file2.write(" ".join(filtered_words))

print(f"\nAfter Removing Stop Words:\n{filtered_words}\nLength: {len(filtered_words)}")
