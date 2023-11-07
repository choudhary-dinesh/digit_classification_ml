from flask import Flask,request
from joblib import  load
import numpy as np
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

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
    img1 = input_data['img1']
    img2 = input_data['img2']

    img1 = list(map(float, img1))
    img2 = list(map(float, img2))

    img1 = np.array(img1).reshape(1,-1)
    img2 = np.array(img2).reshape(1,-1)

    model = load("models/svm_gamma:0.01_C:0.1.joblib")
    predicted1 = model.predict(img1)
    predicted2 = model.predict(img2)
    if predicted1[0] == predicted2[0]:
        print("Both images are of same digit")
        return "True"
    else:
        print("Both images are not of same digit")
        return "False"
    
    # return f"{predicted1}, {predicted2}"

