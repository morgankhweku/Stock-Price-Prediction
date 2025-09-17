import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('Google_Stock_Price_Train.csv')
df = data_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = scaler.fit_transform(df)

x_train = []
y_train = []

for i in range(60,1258):
    x_train.append(df[i -60:i,0])
    y_train.append(df[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

#Reshaping
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Building the Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequencess = True))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

#Compiling the model
model.compile(loss = 'mse', optimizer = 'adam')

#Fitting the model
model.fit(x = x_train, y = y_train, epochs =100, batch_size = 32)

#Making the Prediction
data_test = pd.read_csv('Google_Stock_Price_Test.csv')
df_test =data_test.iloc[:,1:2].values

dataset_total = pd.concat((data_train['Open'], data_test['Open']),axis=0)

inputs = dataset_total[len(dataset_total) - len(data_test) - 60: ].values

inputs = inputs.reshape(-1,1)

inputs = scaler.transform(inputs)

x_test = []

for i in range(60,len(inputs)):
    x_test.append(inputs[i-60:i,0])
x_test =np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

prediction = model.predict(x_test)

prediction = scaler.inverse_transform(prediction)

#Visualization Prediction and Actual Stock
plt.plot(df_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(prediction, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Actual Vs Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price(USD)')
plt.legend()
plt.show()

#Evaluating the model
from sklearn.metrics import mean_squared_error,mean_absolute_error

mae = mean_absolute_error(df_test,prediction)
rmse = np.sqrt(mean_squared_error(df_test, prediction))

print("RMSE: ", rmse)
print("MAE: ", mae)