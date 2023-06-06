import pandas as pd 
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app= Flask(__name__)


# load the pickle model

irismodel =pickle.load(open('irismodel.pkl','rb'))

@app.route('/predict', methods=['GET'])
def predict():
    json = request.json
    query_df = pd.DataFrame(json_)
    prediction = irismodel.predict(query_df)
    return jsonify({prediction : list(prediction)})
if __name__=="__main__":
    app.run(debug=True)


