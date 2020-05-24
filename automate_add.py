# %%
def addTopModel(bottom_model, D=4):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    #top_model = Flatten(name = "flatten")(top_model)
    top_model = tf.keras.layers.Dense(D, activation = "relu")(top_model)
    #top_model = Dropout(0.3)(top_model)
    top_model = tf.keras.layers.Dense(1, activation = "relu")(top_model)
    return top_model

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
dataset = pd.read_csv('/python/50_Startups.csv')

dataset.info()


dataset.columns

y = dataset['Profit']


X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]

state= dataset['State']


state= pd.get_dummies(state, drop_first=True )

rd = dataset['R&D Spend']

ad=dataset['Administration']

ms=dataset['Marketing Spend']

X = pd.concat([rd,ad,ms,state] ,  axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)





model = tf.keras.models.load_model('/python/ann_model1.h5')

FC_Head = addTopModel(model)

modelnew = tf.keras.models.Model (inputs=model.input, outputs=FC_Head)

print(modelnew.summary())

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

info=model.fit(X_train,y_train,epochs=30)

y_pred = model.predict(X_test)

# %%
print(y_pred)


# %%
print(y_test)


model.save("/python/ann_model.h5")

accu=np.mean(info.history['accuracy'])

accu = accu * 100
op_file = open("/python/op_file.sh", "w+")
l=[ "%d" %accu," ""2\n"]
op_file.writelines(l)
op_file.close()


# %%


# %%
