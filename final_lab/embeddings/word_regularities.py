import os 
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def main():
    #Create a dictionary/map to store the word embeddings
    embeddings_index = {}
    #Load pre-computed word embeddings
    #These can be dowloaded from https://nlp.stanford.edu/projects/glove/
    #e.g., wget http://nlp.stanford.edu/data/glove.6B.zip
    #Different embeddings sizes are available. Bigger size, more representation power, more computational cost.
    embeddings_size = "300"
    file_path = '../../data'
    with open(os.path.join(file_path, 'glove.6B.'+embeddings_size+'d.txt'), encoding="utf-8") as f:
        #Process file and load into embeddings_index structure
        print('Loading word embeddings...')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    #Compute distances among first X words
    print('Computing pair-wise distances...')
    max_words = 300
    mat = pairwise_distances(list(embeddings_index.values())[:max_words], metric="cosine")

    #Compute the most similar word for every word. To allow argmin, replace self distances with inf
    np.fill_diagonal(mat, np.inf)
    min_0 = np.argmin(mat,axis=0)

    #Save the pairs to a file
    print('Storing most similar pairs to file')
    with open('similarity_pairs_dim'+embeddings_size+'first'+str(max_words)+'.txt', 'w') as f_out:
        for i, item in enumerate(list(embeddings_index.keys())[:max_words]):
            f_out.write(str(item)+' '+str(list(embeddings_index.keys())[min_0[i]])+'\n')

    #Compute embedding of the "king - man + woman = queen" analogy
    print('Computing analogy')
    embedding_analogy = embeddings_index['king'] - embeddings_index['man'] + embeddings_index['woman']
    embedding_analogy = embeddings_index['doctor'] - embeddings_index['man'] + embeddings_index['woman']
    #embedding_analogy = embeddings_index['nurse'] - embeddings_index['woman'] + embeddings_index['man']
    #Find distances with the rest of the words
    analogy_distances = np.empty(len(embeddings_index))
    for i, item in enumerate(embeddings_index.values()):
        analogy_distances[i] = pairwise_distances(embedding_analogy.reshape(1, -1), item.reshape(1, -1))
    #Get top 10 results
    #Get top 10 results in ascending order
    top_10_indices = np.argsort(analogy_distances)[:10]
    top_10_words = [list(embeddings_index.keys())[i] for i in top_10_indices]
    #Compute similarity of the top 10 in the analogy
    similarity_scores = []
    for word in top_10_words:
        cos_sim = np.dot(embeddings_index[word], embedding_analogy) / (np.linalg.norm(embeddings_index[word]) * np.linalg.norm(embedding_analogy))
        similarity_scores.append(cos_sim)
    #Print results
    print('Top 10 words in analogy: {}'.format(top_10_words))
    print('Similarity scores: {}'.format(similarity_scores))

if __name__ == '__main__':
    main()