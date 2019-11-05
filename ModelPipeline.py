##1. not included model optimization method
import numpy as np
import time
import datetime
import os
import configparser
#for scaling and preprocessing
from sklearn import  preprocessing
from sklearn.preprocessing import StandardScaler
#for accuracy
from sklearn.metrics import roc_auc_score
#for randomness
import random
from random import shuffle
from sklearn.cross_validation import StratifiedKFold
#Sklearn models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
#Gradient Boosting models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from rgf.sklearn import RGFClassifier
#Keras neural network
from keras.models import Sequential
from keras.optimizers import SGD,Adam,RMSprop
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils

SEED = 42  # always use a seed for randomized procedures
config=configparser.ConfigParser()
config.read('config.ini')

def load_data(filename,start,end,id_col,label,use_labels=True):
    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open( filename), delimiter=',',
                      usecols=list(range(start, end)), skiprows=1)
    ids = np.loadtxt(open( filename), delimiter=',',
                            usecols=[id_col], skiprows=1)
    if use_labels:
        labels = np.loadtxt(open( filename), delimiter=',', usecols=[label], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return ids,labels, data

def bagged_set(X_t,y_c,model, seed, estimators, xt,select):
    # create array object to hold predictions 
    baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
    #loop for as many times as we want bags
    for n in range (0, estimators):
        if select == 'kerasnn':
            y_c = np_utils.to_categorical(y_c, 2)
        model.fit(X_t,y_c) # fit model
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):           
            baggedpred[j]+=preds[j]
    # divide with number of bags to create an average estimate            
    for j in range (0, len(baggedpred)): 
        baggedpred[j]/=float(estimators)
    # return probabilities            
    return np.array(baggedpred) 
   
# using numpy to print results
def printfilcsve(X, filename,foldername):
    if not os.path.exists(os.getcwd()+'\\'+foldername):
        os.makedirs(foldername)
    np.savetxt(foldername+'//'+filename,X)
    
# compute all pairs of variables 2ways.
def Make_2way(X, Xt):
    columns_length=X.shape[1]
    for j in range (0,columns_length,2):
        for d in range (j+1,columns_length,random.randint(1,5)):  
            print(("Adding columns' interaction %d and %d" % (j, d) ))
            new_column_train=X[:,j]+X[:,d]
            new_column_test=Xt[:,j]+Xt[:,d]    
            X=np.column_stack((X,new_column_train))
            Xt=np.column_stack((Xt,new_column_test))
    return X, Xt
    
# compute all pairs of variables 3ways.
def Make_3way(X, Xt):
    columns_length=X.shape[1]
    for j in range (columns_length):
        for d in range (j+1,columns_length):  
            for m in range (d+1,columns_length):              
                print("Adding columns' interaction %d and %d and %d" % (j, d, m) )
                new_column_train=X[:,j]+X[:,d]+X[:,m]
                new_column_test=Xt[:,j]+Xt[:,d]+Xt[:,m]      
                X=np.column_stack((X,new_column_train))
                Xt=np.column_stack((Xt,new_column_test))
    return X, Xt
	
def apply_oneHot_encoder(X, X_test):
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder(sparse=False)
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)
    X_test = encoder.transform(X_test)
    return X,X_test
	
def scaled_data(X, X_test,scales):
    if scales=='standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
    if scales=='normal':
        X = preprocessing.normalize(X)
        X_test = preprocessing.normalize(X_test)
    if scales=='scale':
        X = preprocessing.scale(X)
        X_test = preprocessing.scale(X_test)
    return X,X_test
	
def train_bagged_model(X,y,model,bagging,n,select):
    #for calculating the timing.
    start = time.process_time()
    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ]

    # === training & metrics === #
    mean_auc = 0.0
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creating and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
        
        # hyperparameter optimization methods here
        
        # train model and make predictions 
        preds=bagged_set(X_train,y_train,model, SEED , bagging, X_cv,select)   
        
        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
        mean_auc += roc_auc
        
        no=0
        for real_index in test_index:
            train_stacker[real_index]=(preds[no])
            no+=1
        i+=1
    mean_auc/=n
    elapsed_time = time.process_time() - start
    return train_stacker,mean_auc,elapsed_time

def model_selection(select,X):
    if select=='xgb':
        filename="main_xgboost" # xgboost model
        model = XGBClassifier(num_round=1000 ,nthread=25,  eta=0.02, gamma=1,max_depth=20, min_child_weight=0.1, subsample=0.9, colsample_bytree=0.5,objective='binary:logistic',seed=1)
    elif select=='rgf':
        filename="main_rgf" # regression greedy forest model
        model = RGFClassifier(max_leaf=400,algorithm="RGF",test_interval=150, loss="LS")
    elif select=='logit':
        filename="main_logit" # LogisticRegression model
        model = LogisticRegression(C=0.7, penalty="l2")
    elif select=='knn':
        filename="main_knn" # KNearestNeighbor model
        model = KNeighborsClassifier(n_neighbors = 7)
    elif select=='xtree':
        filename="main_xtratree" # extratree model
        model=ExtraTreesClassifier(n_estimators=10000, criterion='entropy', max_depth=9,  min_samples_leaf=1,  n_jobs=30, random_state=1) 
    elif select=='rfc':
        filename="main_rfc" # RandomForest model
        model = RandomForestClassifier(random_state = 13)
    elif select=='cat':
        filename="main_catboost" # Catboost model
        model=CatBoostClassifier(iterations=80, depth=3, learning_rate=0.1, loss_function='Logloss')
    elif select=='svm':
        filename="main_svm" # Support vector machine model
        model = svm.SVC(kernel='linear',probability=True) 
    elif select=='kerasnn':
        filename="main_kerasnn" # Keras Neural network model
        model = keras_network(X)
    else:
        filename="main_lgmboost" # light gradient boosting model
        model=lgb.LGBMClassifier(num_leaves=150,objective='binary',max_depth=6,learning_rate=.01,max_bin=400,auc='binary_logloss')
    return filename,model

def keras_network(X):
	
    fir_activation=config.get('KerasModel', 'fir_activation')	#First activation of the model
    last_activation=config.get('KerasModel', 'last_activation')	#Last activation of the model
    out_dim=int(config.get('KerasModel', 'out_dim'))		#output dimension of model
    last_out_dim=int(config.get('KerasModel', 'last_out_dim'))	#last output dimension
    fir_drop=float(config.get('KerasModel', 'fir_drop'))		#first dropout
    last_drop=float(config.get('KerasModel', 'last_drop'))	#last dropout
    learn_rate=float(config.get('KerasModel', 'learn_rate')) 	#Learning rate of model
	
    model = Sequential()
    model.add(Dense(output_dim=out_dim, input_dim=X.shape[1], init='normal'))
    model.add(Activation(fir_activation))
    model.add(Dropout(fir_drop))
    model.add(Dense(output_dim=out_dim, input_dim=out_dim, init='normal'))
    model.add(Activation(fir_activation))
    model.add(Dropout(last_drop))
    model.add(Dense(output_dim=last_out_dim, input_dim=out_dim, init='normal'))
    model.add(Activation(last_activation))
    model.compile(optimizer=Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
	
def process_features(X,X_test,scales,mk_2way,mk_3way,one_hot,scale_data):
    if mk_2way:
        X,X_test= Make_2way(X, X_test)	# add 2-way interactions
        print("2way Applied")
    if mk_3way:
        X,X_test= Make_3way(X, X_test)	# add 3-way interactions
        print("3way Applied")
    if one_hot:
        X,X_test=apply_oneHot_encoder(X, X_test)	# === one-hot encoding === #
        print("one-hot encoding Applied")
    if scale_data:
        X,X_test=scaled_data(X, X_test,scales)	#scale data standard,normal,scale
        print("Scaling Applied")
    return X, X_test

def main():

    # === load configuration settings === #
    print("loading the configuration settings")
        
    foldername=config.get('General', 'foldername')		#folder name for storing trained model.
    index_column=int(config.get('FileRelated', 'index_column'))	#index or id column number
    label_column=int(config.get('FileRelated', 'label_column'))	#label column number
    start=int(config.get('FileRelated', 'start'))		#feature starting column
    end=int(config.get('FileRelated', 'end'))			#feature ending column
    bagging=int(config.get('FileRelated', 'bagging')) 		# number of models trained with different seeds
    num_folds =int(config.get('FileRelated', 'num_folds'))  	# number of folds in stratified cv
        
    mk_2way=bool(config.get('PreProcess', 'mk_2way'))		#make_2_way interaction of features
    mk_3way=bool(config.get('PreProcess', 'mk_3way'))		#make_3_way interaction of features
    one_hot=bool(config.get('PreProcess', 'one_hot'))		#apply one_hot_encode to sparse features
    scale_data=bool(config.get('PreProcess', 'scale_data'))	#different scaling of data.
    scales = config.get('PreProcess', 'scales')			#type of scaling want to use

    # === Performance file === #
    perform=open('performance.csv','a')
    if sum(1 for line in open('performance.csv'))<2:
        perform.write('timestamp,bagging,num_folds,mk_2way,mk_3way,one_hot,scale_data,scales,model,elapsed_time,mean_auc'+'\n')

    # === load data in memory === #
    print("loading data")
    id, y, X = load_data('train.csv',start,end,index_column,label_column)
    id_test, y_test, X_test = load_data('test.csv',start,end,index_column,label_column, use_labels=False)
        
    # === preprocessing the features === #
    print("preprocessing data")
    X,X_test = process_features(X,X_test,scales,mk_2way,mk_3way,one_hot,scale_data)
        
    # === model selection === #
    print("model selection")
    model_select=config.get('Model', 'model_select').split(',')	#different type of models.
        
    for select in model_select:
        print("%s model is running.." %select)
        filename,model = model_selection(select,X)
        if select=='kerasnn':
            bagging=1
        train_stacker,mean_auc,elapsed_time = train_bagged_model(X,y,model,bagging,num_folds,select)
        print("time spent to train the model: %f" % elapsed_time)
        print((" Average AUC: %f" % (mean_auc) ))
        
        timestamp=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        perform.write(timestamp+','+str(bagging)+','+str(num_folds)+','+str(mk_2way)+','+str(mk_3way)+','+str(one_hot)+','+str(scale_data)+','+scales+','+select+','+str(elapsed_time)[:6]+','+str(mean_auc)[:6]+'\n')

        print (" printing train datasets ")
        printfilcsve(np.array(train_stacker), filename + ".train.csv",foldername)          
        # === Predictions === #
        # When making predictions, retrain the model on the whole training set
        preds=bagged_set(X, y, model, SEED, bagging, X_test,select)  
        
        #create submission file 
        printfilcsve(np.array(preds), filename+ ".test.csv",foldername)
    perform.close()

if __name__ == '__main__':
    main()
