import os
import gensim
from sklearn.decomposition import PCA
import numpy as np

'''
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
def preProcessText(text):
    tokens = nltk.word_tokenize(text)
    #lowercase all words and remove words length that is less than 3
    lowerTokens = [t.lower() for t in tokens if len(t) > 3]
    #remove stopwords
    stopword = nltk.corpus.stopwords.words('english')
    tokensStops = [each for each in lowerTokens if not each in stopword]
    #lemmatization and stemming
    port = PorterStemmer()
    wnl = WordNetLemmatizer()
    tokenLemmatized = [wnl.lemmatize(i) for i in tokensStops]
    #tokenStemmed = [port.stem(i) for i in tokenLemmatized]
    return tokenLemmatized
'''

#====================================================================================================
#main

#convert word2vec model to numpy array
model = gensim.models.Word2Vec.load("word2vec.model")
model.init_sims()
vec = np.zeros((len(model.wv.vocab),len(model.wv['the'])))
i = 0
for word in model.wv.vocab:
    # print(word)
    vec[i] = model.wv[word]
    i += 1
    print(word)

print(vec.shape)
print(vec)

#PCA
#documentation sklean pca (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
pca = PCA(n_components=10)
pca.fit(vec)

print(pca.explained_variance_ratio_)