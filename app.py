from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("app.html")


@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1, -1)
    prediction=model.predict(final_features)
    return render_template('app.html',prediction_text='Your body performance class is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')