from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/')  # Corrected: Indentation for route decorator
def main_page():
    return render_template('index.html')


@app.route('/submit_form', methods=['POST'])
def submit_form():
    input_features = request.form.to_dict( )

    input_data = [list(input_features.values( ))]  # Collect all features in a single row
    input_dataframe = pd.DataFrame(input_data, columns=input_features.keys( ))


    le = LabelEncoder( )
    cate_column = input_dataframe.select_dtypes(exclude=np.number).columns
    input_dataframe[cate_column] = input_dataframe[cate_column].apply(le.fit_transform)

    input_array = input_dataframe.to_numpy( )

    if input_array.size == 0:
        return 'input array size is 0'

    model = joblib.load('churn_model.pkl')
    prediction = model.predict(input_array)

    churn_label = 'Churn' if prediction == 1 else 'Not Churn'
    return render_template('result.html', prediction=churn_label)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
