import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
os.chdir("C:/Users/karth/Desktop/USC/Sem-2/EE559/Project")

input_data_train = pd.read_csv('./aps_failure_training_set_SMALLER.csv', skiprows=20,keep_default_na=False)
input_data_test = pd.read_csv('./aps_failure_test_set.csv', skiprows=20,keep_default_na=False)

display(input_data_train.head(3))
display(input_data_train.tail(3)) 

# replacing 'na' strings
input_data_train.replace('na','-1', inplace=True)
input_data_test.replace('na','-1', inplace=True)
display(input_data_test.tail(3))

#categorical encoding
input_data_train['class'] = pd.Categorical(input_data_train['class']).codes
input_data_test['class'] = pd.Categorical(input_data_test['class']).codes

print(['neg', 'pos'])
print(np.bincount(input_data_train['class'].values))
print(np.bincount(input_data_test['class'].values))

plt.close('all')
bins = np.bincount(input_data_train['class'].values)
plt.bar([0,1], bins, color='black')
plt.xticks([0,1])
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Histogram of target classes [train set]')
plt.show()
