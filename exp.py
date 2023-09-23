"""
Author: Dinesh Kumar, M22AIE227
digit classification model code for mlops 

"""

#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import get_list_of_param_comination, tune_hparams,image_resize_fn
 


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()
###for quiz1 
# print("total no of images in datasat",y.shape[0])
# print("size of each image in datasat ", X[0].shape)

#2.Sanity check of data

################################################################################################
#taking different combinations of train dev and test and reporting results
#3. Spliting the data
test_size =  0.2
dev_size = 0.1
img_sizes = [(4,4), (6,6), (8,8)]
X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)  
for i_size in img_sizes:

        #################################################################################################
        ##rescaling images
        X_train = image_resize_fn(list(X_train),i_size)
        X_test= image_resize_fn(list(X_test),i_size)
        X_dev =image_resize_fn(list(X_dev),i_size)
        # print(X_train.shape, X_test.shape, X_dev.shape)
        #4. Preprocessing the data
        X_train = data_preprocessing(X_train)
        X_test= data_preprocessing(X_test)
        X_dev =data_preprocessing(X_dev)
        # print(X_train.shape, X_test.shape, X_dev.shape)

                                                                                                                    
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
        print('img_size',i_size,'test_size=',0.2,' dev_size=',0.1,' train_size=',0.7,' train_acc=',train_accuracy,' dev_acc',best_val_accuracy,' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
       
