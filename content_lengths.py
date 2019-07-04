import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("../dissim/Sorcero/BBC_2018Q1CSV.csv",usecols=["body"])
s = []
for idx, doc in enumerate(df.body):
    doc_new = [lemmatizer.lemmatize(word) for word in word_tokenize(doc) if word not in stop_words]
    s.append([idx, len(doc_new)])
s = np.array(s)
plt.plot(s[:,0], s[:,1])
# dataset = np.array([[idx, len(doc)] for idx,doc in enumerate(df.body)]) 
# plt.plot(dataset[:,0], dataset[:,1])
plt.xlabel("Document series")
plt.ylabel("Content word count")
plt.show()