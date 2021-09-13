
#importing the libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    exp =  int(request.form.get("exp"))
    test =  float(request.form.get("test"))
    iscore =  float(request.form.get("iscore"))
    final_features = [[exp,test,iscore]]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)