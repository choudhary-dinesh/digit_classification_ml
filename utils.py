#imports
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm

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


#model training
def train_model(X_train, y_train, model_params, model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC
    model = clf(**model_params)
    model.fit(X_train, y_train)
    return model
