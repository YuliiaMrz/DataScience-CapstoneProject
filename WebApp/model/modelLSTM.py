from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)

#load and preprocess the dataset
df = pd.read_csv('Google_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)
dataClose = df[['Close']].values

# Define the LSTM model
def create_and_train_model():
    X, y = splitSequence(dataClose)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1])

    model = Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(30, 1)))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=20, verbose=1)
    return model

def splitSequence(seq):
    #declare X and y as empty lists
    X = []
    y = []

    for i in range(len(seq)):
        #get the last index
        stepsIn = i + 30
        stepsOut = stepsIn + 5
        

        #if lastIndex is greater than the length of sequence than break 
        if stepsOut > len(seq):
            break

        #create input and output sequence
        seq_X, seq_y = seq[i:stepsIn], seq[stepsIn:stepsOut]

        #append seq_X, seq_y in X and y list
        X.append(seq_X)
        y.append(seq_y)

        pass

    #convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)

    return X, y
    
model = create_and_train_model()


def predict_next_5_days(data, target_date):
    target_date = pd.to_datetime(target_date, format='%d/%m/%Y')

    if target_date.weekday() >= 5:
        return None, "Target date falls on a weekend. Please choose a weekday."

    if target_date not in df.index:
        return None, f"Target date {target_date.date()} not found in the data. Please provide a valid date."

    target_index = df.index.get_loc(target_date)

    if target_index < 30 or target_index + 5 > len(df):
        return None, "Not enough data to predict for this date."
   
    #prepare the input for prediction
    last_sequence = df['Close'].values[target_index - 30:target_index].reshape(1, 30, 1)
    #predict the next 5 days
    predictions = model.predict(last_sequence, verbose=0)
    
    return predictions, None

#route to render the form
@app.route('/')
def index():
    return render_template('index.html')

#route to handle form submission
@app.route('/', methods=['POST'])
def predict():
    target_date = request.form['date']
    predicted_prices, error = predict_next_5_days(target_date)

    return render_template(
        'index.html',
        predicted_prices=predicted_prices,
        target_date=target_date,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)