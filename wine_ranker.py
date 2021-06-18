import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#first we import in our data set
df = pd.read_csv('C:/Users/jonah/Desktop/programing practice/wine ranking/winequality-red2.csv')
#the columns present are as follows
# fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,total sulfur dioxide, density, pH, sulphates, alcohol, quality
# we will predict the quality based on the rest of the variables using a simple neural network


# we begin by mapping the quality dataset since it is a sparse ranking with no entries for many of the numerical options. 
#I have decided to categorize the quality score into 3 groups: 
#poor for wines which earned scores of 3 or 4. quality=0
#good meaning the central peak of 5 and 6. quality=1
#great being the top rated wines which earned scores of 7 and 8. quality=2

ranking = pd.DataFrame(0,index=np.arange(len(df['quality'])), columns=['rank'])

for i in range(df['quality'].size):
    if df['quality'][i]>6:
        ranking['rank'][i]=2
    elif df['quality'][i]>4: 
        ranking['rank'][i]=1
    else:
        ranking['rank'][i]=1


# I will briefly combine the rankings so that our split into test and trial groups is matched together
df = df.drop(columns='quality')
df = pd.concat([df,ranking],axis=1)

#now we split the data into test data and train data. 70% used for training

train_data, test_data = np.split(df.sample(frac=1, random_state=1729), [int(0.7 * len(df))])


#we seperate back out the rankings columns from the rest of the data to allow for proper fitting
train_sol=train_data[['rank']]
train_data=train_data.drop(columns=['rank'])

test_sol=test_data[['rank']]
test_data=test_data.drop(columns=['rank'])


#setup the Neural Network via Keras
length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(3, activation='softmax'))

#here we choose our optimizer and loss function, as well as the metric to display as each epoch completes
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_sol, epochs=400)

#now we use the test data to see if our model holds up against data it has never seen before
loss_value, accuracy_value = model.evaluate(test_data, test_sol)
print(f'Our test accuracy was {accuracy_value}')