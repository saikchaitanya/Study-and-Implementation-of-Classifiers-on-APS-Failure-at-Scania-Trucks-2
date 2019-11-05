import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Getting data shape and column labels
def get_shape_and_col_labels(df):
    #Get the shape of the data
    num_rows,num_cols = df.shape
    #Get the col labels 
    data_cols = df.columns.values.tolist()
    return num_rows,num_cols,data_cols

# 1. DATA INPUT  (Inputing data training and test) and replacing na by nan
data = pd.read_csv('./aps_failure_training_set_SMALLER.csv')
data = data.replace(['na'], [np.NaN])
test_data = pd.read_csv('./aps_failure_test_set.csv')
test_data = test_data.replace(['na'], [np.NaN])
data['class'] = pd.Categorical(data['class']).codes
test_data['class'] = pd.Categorical(test_data['class']).codes

#Visualizing the two classes

#Printing the count of total negative and positive class values in training and test data
print("Count of negative and posititve values in Train Data:")
print(['neg', 'pos'])
print(np.bincount(data['class'].values))
print("Count of negative and posititve values in Test Data:")
print(['neg', 'pos'])
print(np.bincount(test_data['class'].values))

#Plotting the histogram for the training dataset classes
plt.close('all')
bins = np.bincount(data['class'].values)
plt.bar([0,1], bins, color='black')
plt.xticks([0,1])
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Histogram of target classes [train set]')
plt.show()
#Plotting the histogram for the test dataset classes
plt.close('all')
bins = np.bincount(test_data['class'].values)
plt.bar([0,1], bins, color='black')
plt.xticks([0,1])
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Histogram of target classes [test set]')
plt.show()

# Train data Edit
num_rows,num_cols,data_cols = get_shape_and_col_labels(data)
#Figure out the fraction of na values in the columns
num_nas = data.isnull().sum()
num_nas_vals = num_nas.values
num_nas_vals = num_nas_vals/num_rows
#Removing Columns where the more than 20% of the entrys are 'na'
for col_index in range(num_cols):
    if num_nas_vals[col_index] > 0.2:
        x=data.pop(data_cols[col_index])
        x=test_data.pop(data_cols[col_index])  
#Now that we have removed some of the columns changing the data shape
num_rows,num_cols,data_cols = get_shape_and_col_labels(data)
#Converting all the values to numeric, imputing missing values with median and Taking the Class as Y variable
X = data.drop(['class'], axis=1)
temp = X.apply(pd.to_numeric)
temp = temp.fillna(temp.median()).dropna(axis=1, how='all')
Y = data['class']

#Converting all the values to numeric, imputing missing values with median and Taking the Class as Y_test variable
X_test = test_data.drop(['class'], axis=1)
X_test = X_test.apply(pd.to_numeric)
X_test = X_test.fillna(X_test.median()).dropna(axis=1, how='all')
Y_test = test_data['class']


#------------------------------Model on given data(Just replacing nan with median of column)-------------------------
# LR TRAINING
print("Training 1 logistic regression Started\n")
LogReg1 = LogisticRegression(max_iter=500, multi_class='ovr', solver='liblinear')
LogReg1.fit(temp, Y)
print("Training 1 logistic regression Complete\n")
# LR ASSESSMENT
Y_pred1 = np.array(LogReg1.predict(X_test))
print("Accuracy score for LR prediction on given data:", accuracy_score(Y_test, Y_pred1))
print("F1 score for LR prediction on given data:", f1_score(Y_test, Y_pred1))
confusion_matrix1_LR = confusion_matrix(Y_test, Y_pred1)
print("Confusion matrix for LR prediction on given data:\n", confusion_matrix1_LR)

# SVM TRAINING
print("Training 1 SVM Started\n")
svm1 = SVC(gamma ='auto' , C= 1, kernel='rbf')
svm1.fit(temp,Y)
print("Training 1 SVM Complete\n")
# SVM ASSESSMENT
Y_pred1_SVM = np.array(svm1.predict(X_test))
print("Accuracy score for SVM prediction on given data:", accuracy_score(Y_test, Y_pred1_SVM))
print("F1 score for SVM prediction on given data:", f1_score(Y_test, Y_pred1_SVM))
confusion_matrix1_SVM = confusion_matrix(Y_test, Y_pred1_SVM)
print("Confusion matrix for SVM prediction on given data:\n", confusion_matrix1_SVM)

# Gaussian NB TRAINING
print("Training 1 Gaussion NB Started\n")
NB1 = GaussianNB()
NB1.fit(temp, Y)
print("Training 1 Gaussion NB Complete\n")
# Gaussion NB ASSESSMENT
Y_pred1_NB = np.array(NB1.predict(X_test))
print("Accuracy score for Gaussion NB prediction on given data:", accuracy_score(Y_test, Y_pred1_NB))
print("F1 score for Gaussion NB prediction on given data:", f1_score(Y_test, Y_pred1_NB))
confusion_matrix1_NB = confusion_matrix(Y_test, Y_pred1_NB)
print("Confusion matrix for Gaussion NB prediction on given data:\n", confusion_matrix1_NB)

# KNN TRAINING
print("Training 1 KNN Started\n")
KNN1 = KNeighborsClassifier(n_neighbors=5)
KNN1.fit(temp, Y)
print("Training 1 KNN Complete\n")
# KNN ASSESSMENT
Y_pred1_KNN = np.array(KNN1.predict(X_test))
print("Accuracy score for KNN prediction on given data:", accuracy_score(Y_test, Y_pred1_KNN))
print("F1 score for KNN prediction on given data:", f1_score(Y_test, Y_pred1_KNN))
confusion_matrix1_KNN = confusion_matrix(Y_test, Y_pred1_KNN)
print("Confusion matrix for KNN prediction on given data:\n", confusion_matrix1_KNN)

# ANN TRAINING
print("Training 1 ANN Started\n")
ANN1 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(5, 2), random_state=3)
ANN1.fit(temp, Y)
print("Training 1 ANN Complete\n")
# ANN ASSESSMENT
Y_pred1_ANN = np.array(ANN1.predict(X_test))
print("Accuracy score for ANN prediction on given data:", accuracy_score(Y_test, Y_pred1_ANN))
print("F1 score for ANN prediction on given data:", f1_score(Y_test, Y_pred1_ANN))
confusion_matrix1_ANN = confusion_matrix(Y_test, Y_pred1_ANN)
print("Confusion matrix for ANN prediction on given data:\n", confusion_matrix1_ANN)


#---------------------------------------Normalised and PCA model------------------------------------------------------------

#  NORMILIZATION and PCA on thetraining and test data
scaler = StandardScaler()
scaler.fit(temp)
temp = scaler.transform(temp)
temp = pd.DataFrame(temp)
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

pca = PCA(0.98)
pca.fit(temp)
N = pca.n_components_
print("PCA: ", N)
temp = pd.DataFrame(pca.transform(temp))
pca = PCA(N)
pca.fit(X_test)
print("TEST_PCA: ", pca.n_components_)
X_test = pd.DataFrame(pca.transform(X_test))

#Visualizing the correlation of the train and test sets using a heat map
df_x=temp.iloc[:,0:10]
plt.figure(figsize=(50,50));
a = sns.heatmap(df_x.corr(), annot=True, annot_kws={"size": 20});
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=60)
plt.show();
plt.title('Correlation Heat Map for Training Data for 10 columns')

df_y=X_test.iloc[:,0:10]
plt.figure(figsize=(50,50));
a = sns.heatmap(df_y.corr(), annot=True, annot_kws={"size": 20});
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=60)
plt.show();
plt.title('Correlation Heat Map for Test Data for 10 columns')

# VALIDATION, TRAINING DATA split
X_train, X_validation, Y_train, Y_validation = train_test_split(temp, Y, test_size=0.2, random_state=0)
DF = pd.concat([X_train, Y_train], axis=1)
Y_train = Y_train.rename(columns={'class': 'Flag'})
print("Number of data samples in Training: ", len(Y_train))
print("Number of data samples in Validation: ", len(Y_validation))


#----------------------------------Training and testing with validation------------------------------------------------

# TRAINING LOGISTIC REGRESSION
print("Training 2 logistic regression Started \n")
LogReg2 = LogisticRegression(max_iter=500, multi_class='ovr', solver='liblinear')
LogReg2.fit(X_train, Y_train)
print("Training 2 logistic regression Complete\n")
# ASSESSMENT
Y_pred2 = np.array(LogReg2.predict(X_validation))
print("Accuracy score for LR prediction on normalized and PCA data:", accuracy_score(Y_validation, Y_pred2))
print("F1 score for LR prediction on normalized and PCA  data:", f1_score(Y_validation, Y_pred2))
confusion_matrix2 = confusion_matrix(Y_validation, Y_pred2)
print("Confusion matrix for LR prediction on normalized and PCA  data:\n",confusion_matrix2)

# SVM TRAINING
print("Training 2 SVM Started\n")
svm2 = SVC(gamma ='auto' , C= 1, kernel='rbf')
svm2.fit(X_train,Y_train)
print("Training 2 SVM Complete\n")
# SVM ASSESSMENT
Y_pred2_SVM = np.array(svm2.predict(X_validation))
print("Accuracy score for SVM prediction on normalized and PCA data:", accuracy_score(Y_validation, Y_pred2_SVM))
print("F1 score for SVM prediction on normalized and PCA data:", f1_score(Y_validation, Y_pred2_SVM))
confusion_matrix2_SVM = confusion_matrix(Y_validation, Y_pred2_SVM)
print("Confusion matrix for SVM prediction on normalized and PCA data:\n", confusion_matrix2_SVM)

# Gaussian NB TRAINING
print("Training 2 Gaussion NB Started\n")
NB2 = GaussianNB()
NB2.fit(X_train, Y_train)
print("Training 2 Gaussion NB Complete\n")
# Gaussion NB ASSESSMENT
Y_pred2_NB = np.array(NB2.predict(X_validation))
print("Accuracy score for Gaussion NB prediction on normalized and PCA data:", accuracy_score(Y_validation, Y_pred2_NB))
print("F1 score for Gaussion NB prediction on normalized and PCA data:", f1_score(Y_validation, Y_pred2_NB))
confusion_matrix2_NB = confusion_matrix(Y_validation, Y_pred2_NB)
print("Confusion matrix for Gaussion NB prediction on normalized and PCA data:\n", confusion_matrix2_NB)

# KNN TRAINING
print("Training 2 KNN Started\n")
KNN2 = KNeighborsClassifier(n_neighbors=5)
KNN2.fit(X_train, Y_train)
print("Training 2 KNN Complete\n")
# KNN ASSESSMENT
Y_pred2_KNN = np.array(KNN2.predict(X_validation))
print("Accuracy score for KNN prediction on normalized and PCA data:", accuracy_score(Y_validation, Y_pred2_KNN))
print("F1 score for KNN prediction on normalized and PCA data:", f1_score(Y_validation, Y_pred2_KNN))
confusion_matrix2_KNN = confusion_matrix(Y_validation, Y_pred2_KNN)
print("Confusion matrix for KNN prediction on normalized and PCA data:\n", confusion_matrix2_KNN)

# ANN TRAINING
print("Training 2 ANN Started\n")
ANN2 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(5, 2), random_state=3)
ANN2.fit(X_train, Y_train)
print("Training 2 ANN Complete\n")
# ANN ASSESSMENT
Y_pred2_ANN = np.array(ANN2.predict(X_validation))
print("Accuracy score for ANN prediction on normalized and PCA data:", accuracy_score(Y_validation, Y_pred2_ANN))
print("F1 score for ANN prediction on normalized and PCA data:", f1_score(Y_validation, Y_pred2_ANN))
confusion_matrix2_ANN = confusion_matrix(Y_validation, Y_pred2_ANN)
print("Confusion matrix for ANN prediction on normalized and PCA data:\n", confusion_matrix2_ANN)


#-----------------------------------UNDERSAMPLING training data model-------------------------------------------------------

numberofrecords_pos = len(DF[DF['class'] == 1])
pos_indices = np.array(DF[DF['class'] == 1].index)
neg_indices = np.array(DF[DF['class'] == 0].index)
print(len(pos_indices), len(neg_indices))
random_neg_indices = np.random.choice(neg_indices, numberofrecords_pos, replace=False)
random_neg_indices = np.array(random_neg_indices)
under_sample_indices = np.concatenate([pos_indices, random_neg_indices])
under_sample_data = DF.loc[under_sample_indices, :]
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'class']
Y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'class']
print("Percentage Neg after under sampling: ", len(under_sample_data[under_sample_data['class'] == 0]) / len(under_sample_data))
print("Percentage Pos after under sampling: ", len(under_sample_data[under_sample_data['class'] == 0]) / len(under_sample_data))
print("Total number of data points: ", len(under_sample_data))

###################### Equally distributing ################################
print('Distribution of the Classes ')
print(under_sample_data['class'].value_counts()/len(under_sample_data))
sns.countplot('class', data=under_sample_data)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

# TRAINING LOGISTIC REGRESSION
print("Training 3 logistic regression Started \n")
LogReg3 = LogisticRegression(max_iter=500, C=0.001, penalty='l2', multi_class = 'ovr', solver='liblinear')
LogReg3.fit(X_undersample, Y_undersample.values.ravel())
print("Training 3 logistic regression Completed \n")
# ASSESSMENT
Y_pred3 = np.array(LogReg3.predict(X_validation))
print("Accuracy score for LR prediction on Undersampled, normalized and PCA data:",accuracy_score(Y_validation,Y_pred3))
print("F1 score for LR prediction on Undersampled, normalized and PCA data:",f1_score(Y_validation, Y_pred3)) 
confusion_matrix3 = confusion_matrix(Y_validation, Y_pred3)
print("Confusion matrix for LR prediction on Undersampled, normalized and PCA data:\n",confusion_matrix3)

# SVM TRAINING
print("Training 3 SVM Started\n")
svm3 = SVC(gamma ='auto' , C= 1, kernel='rbf')
svm3.fit(X_undersample, Y_undersample.values.ravel())
print("Training 3 SVM Complete\n")
# SVM ASSESSMENT
Y_pred3_SVM = np.array(svm3.predict(X_validation))
print("Accuracy score for SVM prediction on Undersampled, normalized and PCA data:", accuracy_score(Y_validation, Y_pred3_SVM))
print("F1 score for SVM prediction on Undersampled, normalized and PCA data:", f1_score(Y_validation, Y_pred3_SVM))
confusion_matrix3_SVM = confusion_matrix(Y_validation, Y_pred3_SVM)
print("Confusion matrix for SVM prediction on Undersampled, normalized and PCA data:\n", confusion_matrix3_SVM)

# Gaussian NB TRAINING
print("Training 3 Gaussion NB Started\n")
NB3 = GaussianNB()
NB3.fit(X_undersample, Y_undersample.values.ravel())
print("Training 3 Gaussion NB Complete\n")
# Gaussion NB ASSESSMENT
Y_pred3_NB = np.array(NB3.predict(X_validation))
print("Accuracy score for Gaussion NB prediction on Undersampled, normalized and PCA data:", accuracy_score(Y_validation, Y_pred3_NB))
print("F1 score for Gaussion NB prediction Undersampled, normalized and PCA data:", f1_score(Y_validation, Y_pred3_NB))
confusion_matrix3_NB = confusion_matrix(Y_validation, Y_pred3_NB)
print("Confusion matrix for Gaussion NB prediction on Undersampled, normalized and PCA data:\n", confusion_matrix3_NB)

# KNN TRAINING
print("Training 3 KNN Started\n")
KNN3 = KNeighborsClassifier(n_neighbors=5)
KNN3.fit(X_undersample, Y_undersample.values.ravel())
print("Training 3 KNN Complete\n")
# KNN ASSESSMENT
Y_pred3_KNN = np.array(KNN3.predict(X_validation))
print("Accuracy score for KNN prediction on Undersampled, normalized and PCA data:", accuracy_score(Y_validation, Y_pred3_KNN))
print("F1 score for KNN prediction on Undersampled, normalized and PCA data:", f1_score(Y_validation, Y_pred3_KNN))
confusion_matrix3_KNN = confusion_matrix(Y_validation, Y_pred3_KNN)
print("Confusion matrix for KNN prediction on Undersampled, normalized and PCA data:\n", confusion_matrix3_KNN)

# ANN TRAINING
print("Training 3 ANN Started\n")
ANN3 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(5, 2), random_state=3)
ANN3.fit(X_undersample, Y_undersample.values.ravel())
print("Training 3 ANN Complete\n")
# ANN ASSESSMENT
Y_pred3_ANN = np.array(ANN3.predict(X_validation))
print("Accuracy score for ANN prediction on Undersampled, normalized and PCA data:", accuracy_score(Y_validation, Y_pred3_ANN))
print("F1 score for ANN prediction on Undersampled, normalized and PCA data:", f1_score(Y_validation, Y_pred3_ANN))
confusion_matrix3_ANN = confusion_matrix(Y_validation, Y_pred3_ANN)
print("Confusion matrix for ANN prediction on Undersampled, normalized and PCA data:\n", confusion_matrix3_ANN)


#----------------------------------------SMOTE training data model------------------------------------------------------------

os = SMOTE(random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(temp, Y, test_size=0.2, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['class'])
# CHECK SAMPLES SIZES
print("Total number of undersampled data points:", len(os_data_X))
print("Percentage Pos after over sampling: ", len(os_data_y[os_data_y['class'] == 1]) / len(os_data_X))
print("Percentage Neg after over sampling: ", len(os_data_y[os_data_y['class'] == 0]) / len(os_data_X))

###################### Equally distributing ################################
print('Distribution of the Classes ')
print(os_data_y['class'].value_counts()/len(os_data_y))
sns.countplot('class', data=os_data_y)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

# TRAINING LOGISTIC REGRESSION
print("Training 4 logistic regression Started\n")
LogReg4 = LogisticRegression(max_iter=500, multi_class='ovr', solver='liblinear')
LogReg4.fit(os_data_X, os_data_y.values.ravel())
print("Training 4 logistic regression Complete\n")
# ASSESSMENT
Y_pred4 = np.array(LogReg4.predict(X_valid))
print("Accuracy score for LR prediction on Oversampled, normalized and PCA data:",accuracy_score(y_valid, Y_pred4))
print("F1 score for LR prediction on Oversampled, normalized and PCA  data:",f1_score(y_valid, Y_pred4)) 
confusion_matrix4 = confusion_matrix(y_valid, Y_pred4)
print("Confusion matrix for LR prediction on Oversampled, normalized and PCA data:\n",confusion_matrix4)

# SVM TRAINING
print("Training 4 SVM Started\n")
svm4 = SVC(gamma ='auto' , C= 1, kernel='rbf')
svm4.fit(os_data_X, os_data_y.values.ravel())
print("Training 4 SVM Complete\n")
# SVM ASSESSMENT
Y_pred4_SVM = np.array(svm4.predict(X_valid))
print("Accuracy score for SVM prediction on Oversampled, normalized and PCA data:", accuracy_score(y_valid, Y_pred4_SVM))
print("F1 score for SVM prediction on Oversampled, normalized and PCA data:", f1_score(y_valid, Y_pred4_SVM))
confusion_matrix4_SVM = confusion_matrix(y_valid, Y_pred4_SVM)
print("Confusion matrix for SVM prediction on Oversampled, normalized and PCA data:\n", confusion_matrix4_SVM)

# Gaussian NB TRAINING
print("Training 4 Gaussion NB Started\n")
NB4 = GaussianNB()
NB4.fit(os_data_X, os_data_y.values.ravel())
print("Training 4 Gaussion NB Complete\n")
# Gaussion NB ASSESSMENT
Y_pred4_NB = np.array(NB4.predict(X_valid))
print("Accuracy score for Gaussion NB prediction on Oversampled, normalized and PCA data:", accuracy_score(y_valid, Y_pred4_NB))
print("F1 score for Gaussion NB prediction on Oversampled, normalized and PCA data:", f1_score(y_valid, Y_pred4_NB))
confusion_matrix4_NB = confusion_matrix(y_valid, Y_pred4_NB)
print("Confusion matrix for Gaussion NB prediction on Oversampled, normalized and PCA data:\n", confusion_matrix4_NB)

# KNN TRAINING
print("Training 4 KNN Started\n")
KNN4 = KNeighborsClassifier(n_neighbors=5)
KNN4.fit(os_data_X, os_data_y.values.ravel())
print("Training 4 KNN Complete\n")
# KNN ASSESSMENT
Y_pred4_KNN = np.array(KNN4.predict(X_valid))
print("Accuracy score for KNN prediction on Oversampled, normalized and PCA data:", accuracy_score(y_valid, Y_pred4_KNN))
print("F1 score for KNN prediction on Oversampled, normalized and PCA data:", f1_score(y_valid, Y_pred4_KNN))
confusion_matrix4_KNN = confusion_matrix(y_valid, Y_pred4_KNN)
print("Confusion matrix for KNN prediction on Oversampled, normalized and PCA data:\n", confusion_matrix4_KNN)

# ANN TRAINING
print("Training 4 ANN Started\n")
ANN4 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(5, 2), random_state=3)
ANN4.fit(os_data_X, os_data_y.values.ravel())
print("Training 4 ANN Complete\n")
# ANN ASSESSMENT
Y_pred4_ANN = np.array(ANN4.predict(X_valid))
print("Accuracy score for ANN prediction on Oversampled, normalized and PCA data:", accuracy_score(y_valid, Y_pred4_ANN))
print("F1 score for ANN prediction on Oversampled, normalized and PCA data:", f1_score(y_valid, Y_pred4_ANN))
confusion_matrix4_ANN = confusion_matrix(y_valid, Y_pred4_ANN)
print("Confusion matrix for ANN prediction on Oversampled, normalized and PCA data:\n", confusion_matrix4_ANN)



#----------------------------SMOTE training data model + Best Parameters + 10 fold cross_validation------------------------------------------------------------

# TRAINING LOGISTIC REGRESSION
print("Training 5 logistic regression Started\n")
LogRegBest = LogisticRegression(solver='liblinear', max_iter = 500)
lr_parameters = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100]}
LogReg5 = GridSearchCV(LogRegBest, param_grid=lr_parameters, cv=5, verbose=0)
LogReg5.fit(os_data_X, os_data_y.values.ravel())
print("best parameters for logistic regression on smote data",LogReg5.best_params_)
print("Training 5 logistic regression Complete\n")
# ASSESSMENT
Y_pred5 = np.array(LogReg5.predict(X_valid))
print("Accuracy score for LR prediction on Oversampled, normalized and PCA data and best fit parameters:",accuracy_score(y_valid, Y_pred5))
print("F1 score for LR prediction on Oversampled, normalized and PCA  data: and best fit parameters",f1_score(y_valid, Y_pred5)) 
confusion_matrix5 = confusion_matrix(y_valid, Y_pred5)
print("Confusion matrix for LR prediction on Oversampled, normalized and PCA data and best fit parameters:\n",confusion_matrix5)

# SVM TRAINING
print("Training 5 SVM Started\n")
svm_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
svm5 = GridSearchCV(svc, svm_parameters, cv=5)
svm5.fit(os_data_X, os_data_y.values.ravel())
print("best parameters for SVM on smote data",svm5.best_params_)
print("Training 5 SVM Complete\n")
# SVM ASSESSMENT
Y_pred5_SVM = np.array(svm5.predict(X_valid))
print("Accuracy score for SVM prediction on Oversampled, normalized and PCA data and best fit parameters:", accuracy_score(y_valid, Y_pred5_SVM))
print("F1 score for SVM prediction on Oversampled, normalized and PCA data and best fit parameters:", f1_score(y_valid, Y_pred5_SVM))
confusion_matrix5_SVM = confusion_matrix(y_valid, Y_pred5_SVM)
print("Confusion matrix for SVM prediction on Oversampled, normalized and PCA data and best fit parameters:\n", confusion_matrix5_SVM)

# KNN TRAINING
print("Training 5 KNN Started\n")
knn = KNeighborsClassifier(algorithm = 'brute')
knn_parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 20, 30, 40, 50]}]
KNN5 = GridSearchCV(knn, knn_parameters, cv=5, scoring='accuracy')
KNN5.fit(os_data_X, os_data_y.values.ravel())
print("best parameters for KNN on smote data",KNN5.best_params_)
print("Training 5 KNN Complete\n")
# KNN ASSESSMENT
Y_pred5_KNN = np.array(KNN5.predict(X_valid))
print("Accuracy score for KNN prediction on Oversampled, normalized and PCA data and best fit parameters:", accuracy_score(y_valid, Y_pred5_KNN))
print("F1 score for KNN prediction on Oversampled, normalized and PCA data and best fit parameters:", f1_score(y_valid, Y_pred5_KNN))
confusion_matrix5_KNN = confusion_matrix(y_valid, Y_pred5_KNN)
print("Confusion matrix for KNN prediction on Oversampled, normalized and PCA data and best fit parameters:\n", confusion_matrix5_KNN)

# ANN TRAINING
print("Training 5 ANN Started\n")
ann = MLPClassifier(solver='lbfgs', random_state=3)
ann_parameters = {'learning_rate': ["constant", "invscaling", "adaptive"], 'hidden_layer_sizes': [(5,2), (25,2), (50,2)], 'alpha': [10.0 ** x for x in range(-1,2)], 'activation': ["logistic", "relu", "tanh"]}
ANN5 = GridSearchCV(ann, param_grid=ann_parameters, verbose=2, cv=5)
ANN5.fit(os_data_X, os_data_y.values.ravel())
print("best parameters for ANN on smote data",ANN5.best_params_)
print("Training 5 ANN Complete\n")
# ANN ASSESSMENT
Y_pred5_ANN = np.array(ANN5.predict(X_valid))
print("Accuracy score for ANN prediction on Oversampled, normalized and PCA data and best fit parameters:", accuracy_score(y_valid, Y_pred5_ANN))
print("F1 score for ANN prediction on Oversampled, normalized and PCA data and best fit parameters:", f1_score(y_valid, Y_pred5_ANN))
confusion_matrix5_ANN = confusion_matrix(y_valid, Y_pred5_ANN)
print("Confusion matrix for ANN prediction on Oversampled, normalized and PCA data and best fit parameters:\n", confusion_matrix5_ANN)



#---------------------------------Applying the best fit model to the final test set---------------------------------

# ASSESSMENT of Logistic Regression on final Test Data
Y_pred6_lr = np.array(LogReg5.predict(X_test))
print("Accuracy score for LR prediction on final test data using best fit parameters:",accuracy_score(Y_test, Y_pred6_lr))
print("F1 score for LR prediction on final test data using best fit parameters:",f1_score(Y_test, Y_pred6_lr)) 
confusion_matrix6_lr = confusion_matrix(Y_test, Y_pred6_lr)
print("Confusion matrix for LR prediction on final test data using best fit parameters:\n",confusion_matrix6_lr)
cm_lr = confusion_matrix6_lr.ravel()
cm_lr = pd.DataFrame(cm_lr.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
total_cost_lr = 10*cm_lr.FP + 500*cm_lr.FN
print("Total Cost for LR prediction on final test data using best fit parameters:\n",total_cost_lr)

# ASSESSMENT of SVM on final Test Data
Y_pred6_svm = np.array(svm5.predict(X_test))
print("Accuracy score for SVM prediction on final test data using best fit parameters:",accuracy_score(Y_test, Y_pred6_svm))
print("F1 score for SVM prediction on final test data using best fit parameters:",f1_score(Y_test, Y_pred6_svm)) 
confusion_matrix6_svm = confusion_matrix(Y_test, Y_pred6_svm)
print("Confusion matrix for SVM prediction on final test data using best fit parameters:\n",confusion_matrix6_svm)
cm_svm = confusion_matrix6_svm.ravel()
cm_svm = pd.DataFrame(cm_svm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
total_cost_svm = 10*cm_svm.FP + 500*cm_svm.FN
print("Total Cost for SVM prediction on final test data using best fit parameters:\n",total_cost_svm)

# ASSESSMENT of Gaussian NB on final Test Data 
Y_pred4_NB = np.array(NB4.predict(X_test))
print("Accuracy score for Gaussion NB prediction on final test data using best fit parameters:", accuracy_score(Y_test, Y_pred4_NB))
print("F1 score for Gaussion NB prediction on final test data using best fit parameters:", f1_score(Y_test, Y_pred4_NB))
confusion_matrix6_NB = confusion_matrix(Y_test, Y_pred4_NB)
print("Confusion matrix for Gaussion NB prediction on final test data using best fit parameters:\n", confusion_matrix6_NB)
cm_nb = confusion_matrix6_NB.ravel()
cm_nb = pd.DataFrame(cm_nb.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
total_cost_nb = 10*cm_nb.FP + 500*cm_nb.FN
print("Total Cost for Gaussian NB prediction on final test data using best fit parameters:\n",total_cost_nb)

# ASSESSMENT of KNN on final Test Data
Y_pred6_knn = np.array(KNN5.predict(X_test))
print("Accuracy score for KNN prediction on final test data using best fit parameters:",accuracy_score(Y_test, Y_pred6_knn))
print("F1 score for KNN prediction on final test data using best fit parameters:",f1_score(Y_test, Y_pred6_knn)) 
confusion_matrix6_KNN = confusion_matrix(Y_test, Y_pred6_knn)
print("Confusion matrix for KNN prediction on final test data using best fit parameters:\n",confusion_matrix6_KNN)
cm_knn = confusion_matrix6_KNN.ravel()
cm_knn = pd.DataFrame(cm_knn.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
total_cost_knn = 10*cm_knn.FP + 500*cm_knn.FN
print("Total Cost for KNN prediction on final test data using best fit parameters:\n",total_cost_knn)

# ASSESSMENT of ANN on final Test Data 
Y_pred6_ann = np.array(ANN5.predict(X_test))
print("Accuracy score for ANN prediction on final test data using best fit parameters:",accuracy_score(Y_test, Y_pred6_ann))
print("F1 score for ANN prediction on final test data using best fit parameters:",f1_score(Y_test, Y_pred6_ann)) 
confusion_matrix6_ann = confusion_matrix(Y_test, Y_pred6_ann)
print("Confusion matrix for ANN prediction on final test data using best fit parameters:\n",confusion_matrix6_ann)
cm_ann = confusion_matrix6_ann.ravel()
cm_ann = pd.DataFrame(cm_ann.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
total_cost_ann = 10*cm_ann.FP + 500*cm_ann.FN
print("Total Cost for ANN prediction on final test data using best fit parameters:\n",total_cost_ann)
