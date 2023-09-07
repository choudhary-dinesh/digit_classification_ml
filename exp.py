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
from utils import load_dataset, data_preprocessing, train_model, get_list_of_param_comination, tune_hparams
from utils import split_train_dev_test, predict_and_eval, visualize_first_n_sample_prediction, get_classification_report


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()

#2.Sanity check of data

################################################################################################
#3. Spliting the data
# X_train, X_test, y_train, y_test = train_test_spliting(X,y)   
X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=0.2, dev_size=0.25)  


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

print(best_hparams, best_model, best_accuracy)



################################################################################################
#6. Prediction and evaluation on test sat
# test accuracy
test_accuracy, y_pred = predict_and_eval(best_model, X_test, y_test)
print("accuracy of model on test sat is ", test_accuracy)

# Below we visualize the first 4 test samples and show their predicted
# visualize_first_n_sample_prediction(X_test, y_pred, n = 4)

#print classification report
classification_report = get_classification_report(y_test, y_pred)
print(f"Classification report for classifier {best_model}")
print(classification_report)


# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")
# plt.show()




###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
