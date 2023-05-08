import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_word_embeddings(embedding_path):
    word_embeddings = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vector
    return word_embeddings

def load_book_text(book_path):
    with open(book_path, encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess(text, stop_words, punctuation):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]
    return words

def compute_document_embedding(words, word_embeddings):
    word_vectors = []
    for word in words:
        if word in word_embeddings:
            word_vectors.append(word_embeddings[word])
    if len(word_vectors) == 0:
        return None
    else:
        return np.mean(word_vectors, axis=0)

def check_similarity(embedding, word, word_embeddings):
    if word in word_embeddings:
        word_embedding = word_embeddings[word]
        return 1 - cosine(embedding, word_embedding)
    else:
        return None

def main():
    # Load the GloVe word embeddings
    embedding_path = '../../data/glove.6B.100d.txt'
    word_embeddings = load_word_embeddings(embedding_path)
    # Load the stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Load the Lord of the Rings books
    book1_path = '../../data/Fellowship_of_the_Ring_1.txt'
    book2_path = '../../data/Fellowship_of_the_Ring_2.txt'
    book3_path = '../../data/Fellowship_of_the_Ring_3.txt'

    # Load the book text
    book1_text = load_book_text(book1_path)
    book2_text = load_book_text(book2_path)
    book3_text = load_book_text(book3_path)

    # Preprocess the text
    book1_words = preprocess(book1_text, stop_words, punctuation)
    book2_words = preprocess(book2_text, stop_words, punctuation)
    book3_words = preprocess(book3_text, stop_words, punctuation)

    # Compute the document embeddings
    book1_embedding = compute_document_embedding(book1_words, word_embeddings)
    book2_embedding = compute_document_embedding(book2_words, word_embeddings)
    book3_embedding = compute_document_embedding(book3_words, word_embeddings)

    # Check the similarity with words like "happy" and "violence"
    print("Similarity of 'happy' with each part of the book:")
    print("Part 1:", check_similarity(book1_embedding, "happy", word_embeddings))
    print("Part 2:", check_similarity(book2_embedding, "happy", word_embeddings))
    print("Part 3:", check_similarity(book3_embedding, "happy", word_embeddings))

    print("Similarity of 'violence' with each part of the book:")
    print("Part 1:", check_similarity(book1_embedding, "violence", word_embeddings))
    print("Part 2:", check_similarity(book2_embedding, "violence", word_embeddings))
    print("Part 3:", check_similarity(book3_embedding, "violence", word_embeddings))

if __name__ == '__main__':
    main()