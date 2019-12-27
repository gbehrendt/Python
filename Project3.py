# ME:4150 Artificial Intelligence in Engineering
# Project 3

##########################################################################
#                          LOGISTIC REGRESSION                           #
##########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")

#read from excel file
wine = pd.read_excel("wineData.xlsx")

# here is the datastructure of glass
print(wine.head())
          
# statistical properties of each feature
print ('\n','statistics summary')
for x in wine: 
    c=wine[x]
    print(x, ' min:{:.2f}  max:{:.2f}  mean:{:.2f}  std:{:.2f}'.format(c.min(),c.max(),c.mean(),c.std()))

# calculate the size of each class
print ('\n','Number of data samples in each class:')

#for e in {0, 1}:
for e in range (1, 4, 1):
   print('Class ', e, ': ', sum(n == e for n in wine['Class']))
   
# choose 2 features features
X = wine[['Proline','Ash']].values
y_ = wine['Class']

#three classes
y = np.zeros(len(wine))
for i in range (0, len(wine), 1):
    if y_[i] == 1: 
        y[i] = 1
    elif y_[i] == 2:
        y[i] = 2
    else:
        y[i] = 3

colors = ['r', 'b', 'y']
target_names = ['x >= 13.68%', '13.68% > x > 12.5%', 'x < 12.5%']

# visualize the whole dataset
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
for yy in range(1, 4):
    plt.scatter(X[y==yy, 0], X[y==yy, 1], marker='o', c=colors[yy-1], 
                label=target_names[yy-1])
plt.xlabel('Wine Feature Magnesium (x1)', size=15)
plt.ylabel('Wine Feature Phenols (x2)', size=15)
plt.title('Wine Alcohol Content Data', size= 15)
plt.legend(loc = 'best', fontsize='small')
plt.show()

# training/validation/test split 
# test size is 20%; 
# training size is 80%*75% = 60%;
# validation size is 80%*25% = 20%
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_, X_valid_, y_train, y_valid = train_test_split(X_train_,y_train, random_state = 4)

print('\n')
print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the validation set.'.format(len(y_valid)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_valid = scaler.transform(X_valid_)
X_test = scaler.transform(X_test_)


# logistic classifier training
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(C=1.0)   # initialization and configuration

# Logistic classifier training and 5-fold cross-validation
C_array = np.arange(0.1, 5.1, 0.1)
best_score = 0.0

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
for C_log in C_array:
    clf = LogisticRegression(C=C_log, max_iter=1000000) # initialization and configuration
    valid_score = cross_val_score(clf, X_train, y_train, cv=5)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_C = C_log

print("The best C is ", best_C)

# retrain the best model
clf = LogisticRegression(C=best_C, max_iter=1000000)
clf.fit(X_train, y_train)    # training
print('\n')
print('The training score: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))

#Report decision boundary equations
print("\n")
print("Decision boundaries are:")
for w, b in zip(clf.coef_, clf.intercept_):
    print( "({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0".format(b, w[0], w[1]))
    
# plot decision boundaries, the testing set and its prediction
x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5

x_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
y_pred = clf.predict(X_test)

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', 
                c=colors[yy-1], label=target_names[yy-1])

for yy in range(1, 4):
    plt.scatter(X_test[y_pred==yy, 0], X_test[y_pred==yy, 1], marker='+', s=200, 
                c=colors[yy-1], label=target_names[yy-1])

for w, b, color in zip(clf.coef_, clf.intercept_, colors):
    plt.plot(x_plot, -(x_plot * w[0] + b) / w[1], c=color, alpha=0.8, linewidth=3)
    
plt.title('Logistic test data (.) vs prediction (+): \n test score = {:.2f}%'.
          format(clf.score(X_test, y_test)*100), size=15)
plt.xlabel('wine feature Magnesium (x1)', size=15)
plt.ylabel('wine feature Phenols (x2)', size=15)
plt.legend(loc = 'best',fontsize='small')
plt.show()

# decision regions: a contour plot
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=target_names[yy-1])
plt.title('Logistic decision regions and the test set', size=15)
plt.xlabel('wine feature Magnesium (x1)', size=15)
plt.ylabel('wine feature Phenols (x2)', size=15)
plt.legend(loc = 'best',fontsize='small')
plt.show()
print("\n")

# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Logistic Accuracy:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()


# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))


##########################################################################
#                          SVC WITH RBF KERNEL                           #
##########################################################################

print("\n")
print('SVC w/ RBF Kernel')

# training/validation/test split 
# test size is 20%; 
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

print('\n')
print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

# SVC with rbf kerenel classifier training and cross-validation
from sklearn.svm import SVC
best_score = 0.0
gamma_array = np.arange(0.1, 5.1, 0.1)
C_array = np.arange(0.1, 5.1, 0.1)

from sklearn.model_selection import cross_val_score
for gamma_svc in gamma_array:
    for C_svc in C_array:
        clf = SVC(kernel='rbf', random_state=0, gamma=gamma_svc, C=C_svc)
        valid_score = cross_val_score(clf, X_train, y_train, cv=10)
        if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_parameters = {'C':C_svc, 'gamma':gamma_svc}

print('best score: {:.3f}'.format(best_score))
print('best parameters: {}'.format(best_parameters))

# retrain the best model
clf = SVC(kernel='rbf', random_state=0, **best_parameters)
clf.fit(X_train, y_train)

print('Accuracy of rbf SVC classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of rbf SVC classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
print('\n')



# Plot the decision region vs test data
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  # spacing between grid points
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# plot the results in a contour plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(5, 4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



# plot validation set vs its prediction
y_predict = clf.predict(X_test)

plt.scatter(X_test[y_predict==3,0],X_test[y_predict==3,1], marker = '+', c='y', s=200, label='x < 12.5% prediction')
plt.scatter(X_test[y_predict==2,0],X_test[y_predict==2,1], marker = '+', c='b', s=200, label='13.68% > x > 12.5% prediction')
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='x >= 13.68% prediction')
plt.scatter(X_test[y_test==3,0],X_test[y_test==3,1], marker = 'o', c='y', s=30, label='x < 12.5%')
plt.scatter(X_test[y_test==2,0],X_test[y_test==2,1], marker = 'o', c='b', s=30, label='13.68% > x > 12.5%')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='x >= 13.68%')


plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Alcohol Content test data vs rbf SVC \n test score = {:.2f}% (gamma={:.2f} and C={:.2f}) '
          .format((clf.score(X_test, y_test)*100), best_parameters['gamma'], best_parameters['C']), size=15)
plt.xlabel('wine feature Magnesium (x1)', size=15)
plt.ylabel('wine feature Phenols (x2)', size=15)
plt.legend(loc='best', fontsize = 'small', scatterpoints=1, numpoints=1)
plt.show()


# decision regions: a contour plot
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=target_names[yy-1])
plt.title('SVC decision regions and the test set', size=15)
plt.xlabel('wine feature Magnesium (x1)', size=15)
plt.ylabel('wine feature Phenols (x2)', size=15)
plt.legend(loc = 'best',fontsize='small')
plt.show()
print("\n")

# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_predict)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('SVC Accuracy:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()


# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

