import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pickle model
irismodel = pickle.load(open('irismodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a DataFrame from the input values
    query_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Make the prediction using the loaded model
    prediction = irismodel.predict(query_df)
    predicted_class = prediction[0]

    return render_template('index.html', prediction_text=f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    app.run(debug=True)
