from flask import Flask, request, render_template
import datetime
import pandas as pd
from model.modelLSTM import predict_next_5_days

app = Flask(__name__)

#custom filter to format floats
@app.template_filter('currency')
def currency_filter(value):
    try:
        return "${:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return value
    
@app.route("/", methods=['GET', 'POST'])
def home():
    predicted_prices = None
    target_date = None
    error = None

    if request.method == 'POST':
        date_str = request.form.get('date')
        try:
            target_date = datetime.datetime.strptime(date_str, '%d/%m/%Y')
            df = pd.read_csv('Google_dataset.csv')  # Load the dataframe to use for the date index

            #call the prediction function
            predicted_prices, error = predict_next_5_days(df, date_str)

            if error:
                raise ValueError(error)
            
        except ValueError as e:
            #in case of an error, show the error message
            error = str(e)
        except Exception as e:
            #for other exceptions
            error = 'An unexpected error occurred: ' + str(e)
            
    return render_template('index.html', predicted_prices=predicted_prices, target_date=target_date, error=error)
     
if __name__ == '__main__':
    app.run(debug=True)