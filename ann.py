
# %
import pandas as pd

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
X

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
model = Sequential()

# %%
model.add(Dense(units=2 , input_shape=(5,) , activation='relu' ))

# %%
model.add(Dense(units=2 , input_shape=(5,) , activation='relu' ))

# %%
model.add(Dense(units=1 , input_shape=(5,) , activation='relu' ))

# %%
model.summary()

# %%
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

# %%
model.fit(X_train,y_train,epochs=30)

# %%
y_pred = model.predict(X_test)

# %%
print(y_pred)


# %%
print(y_test)

# %
accuracy=((y_test-y_pred)/y_test)*100

print(accuracy)

exit()
