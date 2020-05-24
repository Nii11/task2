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

print(model.summary())

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001),metrics=['accuracy'])

info=model.fit(X_train,y_train,epochs=30)

np.mean(info.history['accuracy'])

y_pred = model.predict(X_test)

# %%
print(y_pred)


# %%
print(y_test)


model.save("/python/ann_model1.h5")
accu=accu*100
op_file = open("/python/op_file.sh", "w+")
l=[ "%d" %accu," ","5\n"]
op_file.writelines(l)
op_file.close()
