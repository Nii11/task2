# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# %%
from keras.models import Model

# %%
dataset = pd.read_csv('/python/50_Startups.csv')

# %%
dataset.info()

# %%
dataset.columns

# %%
y = dataset['Profit']

# %%
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]

# %%
state= dataset['State']

# %%
state= pd.get_dummies(state, drop_first=True )

# %%
type(X)

# %%
type(state)

# %%
rd = dataset['R&D Spend']

# %%
ad=dataset['Administration']

# %%
ms=dataset['Marketing Spend']

# %%
X = pd.concat([rd,ad,ms,state] ,  axis=1)

# %%
#X

# %%
from keras.models import Sequential


# %%
from sklearn.model_selection import train_test_split

# %%
from keras.layers import Dense

# %%
from keras.optimizers import Adam

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# %%
X_train.shape

# %%
model = tf.keras.models.Sequential()


# %%
tf.keras.Model()

# %%
model.add(tf.keras.layers.Dense(units=10, input_shape=(5,) , activation='relu' , name='init1' ))

# %%
model.add(tf.keras.layers.Dense(units=10 , input_shape=(5,) , activation='relu' , name='init2' ))

# %%
model.add(tf.keras.layers.Dense(units=1 , input_shape=(5,) , activation='relu' , name='init3' ))

# %%
print(model.summary())

# %%
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['mape'])

# %%
info = model.fit(X_train,y_train,epochs=30)

# %%
#info.history['mse']

# %%
import numpy as np

# %%
#accu=np.mean(info.history['mse'])
# %%
y_pred = model.predict(X_test)
sop=len(y_pred)
y_pred=y_pred.reshape(sop)
# %%
print(y_pred)

# %
print(y_test)

accu=y_test-y_pred

accu=accu/y_test

accu=accu*100

accu=100-accu

print(accu)

tot_accu=np.mean(accu)

import os

# %%
op_file = open("/python/op_file.sh", "w+")
l=[ "%d" %tot_accu," ""1\n"]
op_file.writelines(l)
op_file.close()
#%%

# %%
model.save('/python/ann_model1.h5')
