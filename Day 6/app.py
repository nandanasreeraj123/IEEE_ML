import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    if(int_features[1]=='Male'):
        int_features[1] = 1
    else:
        int_features[1] = 2
    int_features_new = [int(x) for x in int_features]
    final_features = [np.array(int_features_new)]
    prediction = model.predict(final_features)

    if(prediction[0]==1):
        output = "Yes"
    else:
        output = "No"
    return render_template('index.html', prediction_text= 'Will he/she purchase: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)