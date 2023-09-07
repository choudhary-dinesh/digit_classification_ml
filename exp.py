"""
================================
Recognizing hand-written digits  
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics

#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import get_list_of_param_comination, tune_hparams
 


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()

#2.Sanity check of data

################################################################################################
#3. Spliting the data
X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=0.2, dev_size=0.2)  


#################################################################################################
#4. Preprocessing the data
X_train = data_preprocessing(X_train)
X_test= data_preprocessing(X_test)
X_dev =data_preprocessing(X_dev)

                                                                                                                    
#################################################################################################
#5. Classification model training
#tuning model for gamma value 
gamma_values = [0.0005, 0.001, 0.002, 0.005, 0.010]
C_values = [0.1, 0.2, 0.5, 0.75, 1]
list_of_param_comination = get_list_of_param_comination([gamma_values, C_values],  ['gamma', 'C'])

best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination)

print(best_hparams, best_accuracy)



################################################################################################
#6. Prediction and evaluation on test sat
# test accuracy
test_accuracy, y_pred = predict_and_eval(best_model, X_test, y_test)
print("accuracy of model on test sat is ", test_accuracy)
