import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import numpy as np

# Check if 'punkt' is loaded
try:
    nltk.data.find('tokenizers/punkt')
    print("The 'punkt' tokenizer is already loaded!")
except LookupError:
    print("The 'punkt' tokenizer is not loaded. Downloading now...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords.zip')
    print("The 'stopwords' dataset is already loaded!")
except LookupError:
    print("The 'stopwords' dataset is not loaded. Downloading now...")
    nltk.download('stopwords')

stemmer = PorterStemmer()

def tokenize(s: str) -> list[str]:
    return nltk.word_tokenize(s) 

def stem(word: str) -> str:
    return stemmer.stem(word)

def bag_of_words(tokenized: list[str], all_words: list[str]):
    tokenized_set = [stem(word) for word in tokenized]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for i in range(len(all_words)):
        if all_words[i] in tokenized_set:
            bag[i] = 1.0

    return bag

if __name__ == '__main__':
    print(sorted(stopwords.words('english')))