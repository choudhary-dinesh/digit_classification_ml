#imports
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

#load dataset from sklearn
def load_dataset():
    digit_data = datasets.load_digits()
    X = digit_data.images
    y = digit_data.target
    return X,y


#data preprocessing
def data_preprocessing(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

#spliting data 
def train_test_spliting(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    return  X_train, X_test, y_train, y_test

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=10)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = dev_size, random_state=10)
    return X_train, y_train, X_test, y_test, X_dev, y_dev



#model training
def train_model(X_train, y_train, model_params, model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC
    model = clf(**model_params)
    model.fit(X_train, y_train)
    return model

#prediction and accuracy evaluation
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = round(metrics.accuracy_score(y_test, predicted), 4) * 100
    return accuracy, predicted


#Visualize first n sample and show their prediction
def visualize_first_n_sample_prediction(X_test, y_pred, n = 4):
    _, axes = plt.subplots(nrows=1, ncols=n, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, y_pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

#return classification report
def get_classification_report(y_test, y_pred):
    return metrics.classification_report(y_test, y_pred)