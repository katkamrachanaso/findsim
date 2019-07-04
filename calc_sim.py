import nltk, string, numpy
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# stemmer = nltk.stem.porter.PorterStemmer()
import math
# def idf(n,df):
# 	result = math.log((n+1.0)/(df+1.0)) + 1
# 	return result

import pandas as pd

class DocumentSim():
	
	def __init__(self, **kwargs):
		'''read csv and fetch text body'''
		self.df = pd.read_csv("../dissim/Sorcero/BBC_2018Q1CSV.csv",usecols=["body"])                                                                                                                           
		self.cos_similarity_matrix = None
		self.calc_similarity()


	def calc_similarity(self):
		remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
		lemmer = nltk.stem.WordNetLemmatizer()
		tfidfTran = TfidfTransformer(norm="l2")

		def _LemTokens(tokens):
			return [lemmer.lemmatize(token) for token in tokens]
		def _LemNormalize(text):
			return _LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
		LemVectorizer = CountVectorizer(tokenizer=_LemNormalize, stop_words='english')
		LemVectorizer.fit_transform(self.df.body)

		tf_matrix = LemVectorizer.transform(self.df.body).toarray()
		tfidfTran.fit(tf_matrix)
		tfidf_matrix = tfidfTran.transform(tf_matrix)
		self.cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
		print(self.cos_similarity_matrix)

d = DocumentSim()
s = np.array(d.cos_similarity_matrix)
# np.argsort(s,axis=1) #ascending
# s.sort(axis=1) from dissimilar to similar score