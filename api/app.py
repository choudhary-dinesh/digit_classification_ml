from flask import Flask,request
from markupsafe import escape
import numpy as np
from joblib import  load

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>This is home page for flask app to serve model on azure cloud</p>"

@app.route("/sum/<x>/<y>")
def sum_two_numbers(x,y):
    res = int(x) + int(y)
    return f"sum of {x} and {y} is {res}"

def load_model():
    lr_model_path  =  'models/M22AIE227_LR_solver_lbfgs.joblib'
    lr_model = load(lr_model_path)

    svm_model_path = "models/M22AIE227_svm_gamma_0.001_C_0.1.joblib"
    svm_model = load(svm_model_path)

    tree_model_path = "models/M22AIE227_tree_max_depth_10.joblib"
    tree_model = load(tree_model_path) 

    return lr_model, svm_model, tree_model

lr_model, svm_model, tree_model = load_model()

print("models", lr_model, svm_model, tree_model)


@app.route("/predict/<model_name>", methods = ['POST'])
def predict_fn(model_name):
    input_data = request.get_json()

    img = input_data['image']

    img = list(map(float, img))

    img = np.array(img).reshape(1,-1)
    
    model_type = escape(model_name)

    print("model_type from request : ", model_type)

    if model_type == "LR":
        model = lr_model 
    elif model_type == "svm":
        model = svm_model 
    elif model_type == "tree":
        model = tree_model 
    # model = load("svm_model.joblib")

    predicted = model.predict(img)
    return f"Prediction : uploaded images belongs to number {predicted[0]} category"


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 80)


    
