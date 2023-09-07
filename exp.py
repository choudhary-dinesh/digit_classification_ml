"""
Author: Dinesh Kumar, M22AIE227
digit classification model code for mlops 

"""

#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import get_list_of_param_comination, tune_hparams
 


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()

#2.Sanity check of data

################################################################################################
#taking different combinations of train dev and test and reporting results
#3. Spliting the data
test_size =  [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]
for ts in test_size:
    for ds in dev_size:
        X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=ts, dev_size=ds)  


        #################################################################################################
        #4. Preprocessing the data
        X_train = data_preprocessing(X_train)
        X_test= data_preprocessing(X_test)
        X_dev =data_preprocessing(X_dev)

                                                                                                                    
        #################################################################################################
        #5. Classification model training
        #hyper parameter tuning for gamma and C
        gamma_values = [0.001, 0.002, 0.005, 0.01, 0.02]
        C_values = [0.1, 0.2, 0.5, 0.75, 1]
        list_of_param_comination = get_list_of_param_comination([gamma_values, C_values],  ['gamma', 'C'])
        best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination)

        #get training accuracy of this best model:
        train_accuracy, _ = predict_and_eval(best_model, X_train, y_train)

        ################################################################################################
        #6. Prediction and evaluation on test sat
        # test accuracy
        test_accuracy, _ = predict_and_eval(best_model, X_test, y_test)

        #print for github actions
        print('test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' train_acc=',train_accuracy,' dev_acc',best_val_accuracy,' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
       
