# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:19:02 2020

@author: usama
"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn

df= pd.read_csv('COVIDFakeNEWSDATA.csv')
#df.head()
df=df.dropna()
x= df.drop('outcome',axis=1)
y=df['outcome']
xshape=x.shape
yshape=y.shape

voc_size=7000
messages= x.copy()
messages.reset_index(inplace=True)

nltk.download('stopwords')

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['headlines'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
onehot_repr= [one_hot(words,voc_size)for words in corpus]
sent_length=60
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs)
embedding_vector_features= 45
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
x_final= np.array(embedded_docs)
y_final= np.array(y)

x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.20,random_state=0)
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=35,batch_size=50)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='best')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training Loss', "validation loss"], loc='best')
plt.show()

y_pred= model.predict_classes(x_test)
cm= confusion_matrix(y_test,y_pred)
print(cm)

df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt='g',annot_kws={"size": 16})# font size




ac= accuracy_score(y_test,y_pred)
print(ac)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

model.save('model_tosave.h5')








