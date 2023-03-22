from flask import Flask, request, render_template
import pandas as pd
import yfinance as yf
from prophet import Prophet

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    ticker = request.form['ticker']
    timeframe = request.form['timeframe']
    freq = request.form['freq']
    length = request.form['length']

    # Get the stock data from Yahoo Finance
    data = yf.download(ticker, period=timeframe)

    # Check for missing values
    if data.isnull().values.any():
        data = data.dropna()

    # Prepare the data
    data = data[['Close']]
    data = data.reset_index()
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Create the model
    model = Prophet()
    model.fit(data)

    # Make a prediction
    future = model.make_future_dataframe(periods=int(length), freq=freq)
    forecast = model.predict(future)

    # Format the prediction as a table
    table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(length)).to_html()

    return render_template('prediction.html', table=table)

if __name__ == '__main__':
    app.run(debug=True)
