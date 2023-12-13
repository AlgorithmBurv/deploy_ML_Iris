# from pyexpat import features
import numpy as np
# import pandas as pd
import pickle
from joblib import load
from flask import Flask, render_template, request

app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))
model = load('model.joblib')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():        
    float_features  =[float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    return render_template('index.html', prediction_text = "{}".format(prediction) )


if __name__ == "__main__":
    app.run(debug=True)


