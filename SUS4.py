# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset_stima.csv', delimiter = ",")
#take all the lines, all the columns except for the target one
X=dataset.iloc[:,1:]
X=X.drop('ClientStatus',axis=1).values
#take all the lines and only the target column
y = dataset['ClientStatus'].values


dataset.isnull().values.any()
dataset.columns[dataset.isna().any()].tolist()


#Taking care of the mising data
from sklearn.preprocessing import Imputer
#use the mean (or "most_frequent") of the other elements in the coluomn
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)#provare dopo con mediana
#only on the coluomns where there are missing values
imputer = imputer.fit(X[:, [1,3]])
X[:, [1,3]] = imputer.transform(X[:, [1,3]])


#Ecoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#transform the first column in numbers
qualitative = [0,1,5,7,12,14,18]
for c in qualitative:
    labelencoder_X = LabelEncoder()
    X[:, c] = labelencoder_X.fit_transform(X[:, c])
    
#SMOTE Oversampling
from imblearn.over_sampling import SMOTE
from sklearn.externals.joblib.parallel import _backend

sm = SMOTE(random_state=0, ratio = 1.0)
X, y = sm.fit_sample(X, y)
X[:,[0,1,2,3,4,5,6,7,8,11,12,14,15,16,17,18,19,22,25,26,27]] = np.round(X[:,[0,1,2,3,4,5,6,7,8,11,12,14,15,16,17,18,19,22,25,26,27]])
    

#create dummy variables
onehotencoder = OneHotEncoder(categorical_features = qualitative)
X = onehotencoder.fit_transform(X).toarray()
#now for the y cloumn, we don't have to use dummy variables beacouse it is the indipendent one
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
#scale the X coloumns
sc_X = StandardScaler()
#for the training set we need to fit it, then scale it
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#do I need to scale the dummy variables? 
#It depends: if i scale i will loose interpretation































################
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
##################
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
##################
# Fitting SVM to the Training set
from sklearn.svm import SVC
#you can choose what type of kernel you want
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
##################
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
#rbf is the gaussian kernel
#if you choose a polynomial kernel you can choose the degree
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
###################
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
####################
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
####################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
#n_estimators = number of trees
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#######################
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
######################
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(loss="deviance",learning_rate = .5, 
                                  n_estimators = 100,max_depth=2,min_samples_leaf=15,min_samples_split=40)
#min_samples_leaf=15
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


################################

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#specify the parameters for wich we want to find the optimal values, give different options in dictionaries
parameters = [{'loss':["deviance","exponential"],'learning_rate': [0.1, 0.05, 0.2], 'n_estimators': [100,1000], "max_depth":[1,3,5]}]
#estimator: the model
#param_grid: the list of parameters
#scoring: scoring metric
#cv: number of folds in k-fold cross validation
#n_jobs: -1 gives all the power available of the machine
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
#fit
grid_search = grid_search.fit(X_train, y_train)
#get best accuracy score
best_accuracy = grid_search.best_score_
#best paramesters
best_parameters = grid_search.best_params_



classifier = GradientBoostingClassifier(mettere i parametri)
classifier.fit(X_train, y_train)














#############TEST





TEST = pd.read_csv('dataset_verifica.csv', delimiter = ",")
#take all the lines, all the columns except for the target one
TEST=TEST.iloc[:,3:].values


#Taking care of the mising data
from sklearn.preprocessing import Imputer
#use the mean (or "most_frequent") of the other elements in the coluomn
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)#provare dopo con mediana
#only on the coluomns where there are missing values
imputer = imputer.fit(TEST[:, [1,3]])
TEST[:, [1,3]] = imputer.transform(TEST[:, [1,3]])

#Ecoding categorical data
#transform the first column in numbers
qualitative = [0,1,5,7,12,14,18]
for c in qualitative:
    labelencoder_X = LabelEncoder()
    TEST[:, c] = labelencoder_X.fit_transform(TEST[:, c])
#create dummy variables
onehotencoder = OneHotEncoder(categorical_features = qualitative)
TEST = onehotencoder.fit_transform(TEST).toarray()

# Feature Scaling
#scale the X coloumns
sc_X = StandardScaler()
#for the training set we need to fit it, then scale it
TEST = sc_X.fit_transform(TEST)

#classifier.fit(X, y)
# Predicting the Test set results
y_pred = classifier.predict(TEST)

#create txt
np.savetxt("output.txt", y_pred, delimiter= "\n", fmt = "%i")