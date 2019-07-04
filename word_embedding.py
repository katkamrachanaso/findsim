from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')

pattern = r"""(?x)          
     (?:[A-Z]\.)+           # abbreviations 
     |\d+(?:\.\d+)?%?       # numbers, currency and percentages 
     |\w+(?:[-']\w+)*       # words with optional internal hyphens/apostrophe 
    """

tokenizer = RegexpTokenizer(pattern)

# define training data
df = pd.read_csv("../dissim/Sorcero/BBC_2018Q1CSV.csv",usecols=["body"])                                                                                                                           
s = []
doc_set = []
for idx, doc in enumerate(df.body):
    doc_new = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(doc) if word not in stop_words]
    doc_set.append(doc_new)
    s.append([idx, len(doc_new)])
s = np.array(s)

# train model
model = Word2Vec(doc_set, min_count=5)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# print(result)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1], color='red')
words = list(model.wv.vocab)
w = words[:50]
for i, word in enumerate(w):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()