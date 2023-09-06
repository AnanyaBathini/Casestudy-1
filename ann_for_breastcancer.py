
#ARTIFICIAL NEURAL NETWORKS (ANN)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
#since the o/p is like a person having breastcancer or not so its binary type of output
 #for this in artificial neural netwrok we need to use sigmoid function as activation unction
 #binary cross entropy as loss function
df=pd.read_csv("/content/data.csv")
df.head()
df.tail()
df.info()
#removing uncanned rows orr columns from that dataset
df.drop(columns=['Unnamed: 32','id'],axis=1,inplace=True)
df.info()
df.describe()
#Splitting the data
x=df.drop(columns=['diagnosis'],axis=1)
y=df['diagnosis']
print("Shape of X:",x.shape)
print("Shape of Y:",y.shape)
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
scaler =MinMaxScaler()
x= scaler.fit_transform(x)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
print("Shape of X train:",x_train.shape)
print("Shape of X test:",x_test.shape)
print("Shape of Y train:",y_train.shape)
print("Shape of Y test:",y_test.shape)
mmodell=tf.keras.models.Sequential()
from keras.layers import Dense,Dropout
mmodell.add(Dense(16,activation='relu',input_dim=30))
mmodell.add(Dropout(0.1))
mmodell.add(Dense(16,activation='relu'))
mmodell.add(Dropout(0.1))
mmodell.add(Dense(1,activation='sigmoid'))
mmodell.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
mmodell.summary()
history=mmodell.fit(x_train,y_train,epochs=150,validation_split=0.2)
