# %%
def addTopModel(bottom_model, D=4):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
#   # top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(1, activation = "relu")(top_model)
    return top_model

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
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





model = keras.models.load_model('/python/ann_model')

FC_Head = addTopModel(model)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

model.fit(X_train,y_train,epochs=30)

y_pred = model.predict(X_test)

# %%
print(y_pred)


# %%
print(y_test)


model.save("/python/ann_model")



# %%


# %%
