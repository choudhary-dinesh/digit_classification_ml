from flask import Flask,request

import numpy as np
from joblib import  load

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>This is home page for flask app to serve model on azure cloud</p>"

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'


@app.route("/sum/<x>/<y>")
def sum_two_numbers(x,y):
    res = int(x) + int(y)
    return f"sum of {x} and {y} is {res}"


@app.route("/predict", methods = ['POST'])
def predict_fn():
    input_data = request.get_json()

    img = input_data['image']

    img = list(map(float, img))

    img = np.array(img).reshape(1,-1)

    model = load("svm_model.joblib")
    predicted = model.predict(img)
    return f"Prediction : uploaded images belongs to number {predicted[0]} category"


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 80)


    
