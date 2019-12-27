# import pandas library
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")

#read from excel file
concrete = pd.read_excel("concrete.xls")

# here is the datastructure of glass
print(concrete.head())
          
# statistical properties of each feature
print ('\n','statistics summary')
for x in concrete: 
    c=concrete[x]
    print(x, ' min:{:.2f}  max:{:.2f}  mean:{:.2f}  std:{:.2f}'.format(c.min(),c.max(),c.mean(),c.std()))

# calculate the size of each class
print ('\n','Number of data samples in each class:')

#for e in {0, 1}:
for e in range (0, 2, 1):
   print('Class ', e, ': ', sum(n == e for n in concrete['Class']))

# visualization 
X = concrete[['cement', 'BFS', 'Fash', 'Water']]
y_ = concrete['Cstr']
y = y_ > 35  # compressive strength larger than 35MPa is the true class

print('\n')
print('The class of compressive strength larger than 35MPa has', sum(e==1 for e in y), 'data samples') #yellow
print('The class of compressive strength smaller than 35MPa has', sum(e==0 for e in y), 'data samples') #purple

# Examine the dataset by visualization 
import matplotlib.pyplot as plt

Axes=pd.plotting.scatter_matrix(X, c= y, marker = 'o', s=40, figsize=(12,12))
[plt.setp(item.yaxis.get_label(), 'size', 20) for item in Axes.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 20) for item in Axes.ravel()]
plt.show()

# choose FA and CA only
X = concrete[['FA', 'CA']].values

# visualize the whole dataset
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X[y==True,0],X[y==1,1], marker = 'o', c='r', label='Greater than 35 MPa')
plt.scatter(X[y==False,0],X[y==0,1], marker = 'o', c='b', label='Less than 35 MPa')
plt.xlabel('concrete feature FA (x1)', size=15)
plt.ylabel('concrete feature CA (x2)', size=15)
plt.title('Compressive Strength of Concrete Data', size= 15)
plt.legend(loc='best', fontsize='large')
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

# Standardization and plot the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_valid = scaler.transform(X_valid_)
X_test = scaler.transform(X_test_)

# visualize the training set only

x1_min, x1_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
x2_min, x2_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_train[y_train==True,0],X_train[y_train==1,1], marker = 'o', c='r', label='Greater than 35 MPa')
plt.scatter(X_train[y_train==False,0],X_train[y_train==0,1], marker = 'o', c='b', label='Less than 35 MPa')
plt.xlabel('concrete feature FA (x1)', size=15)
plt.ylabel('concrete feature CA (x2)', size=15)
plt.title('Compressive Strength of Concrete Training Data', size= 15)
plt.legend(loc='best', fontsize='large')
plt.show()

# logistic classifier training with grid search
C_array = np.arange(0.01, 5.01, 0.01)
best_score = 0.0

from sklearn.linear_model import LogisticRegression

for C_logistic in C_array:
    clf = LogisticRegression(C=C_logistic)
    clf.fit(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    if valid_score > best_score:
        best_score = valid_score
        best_C = C_logistic

print('The best validation score: {:.2f}%'.format(best_score*100))
print('The best parameter C: {}'.format(best_C))

logclf = LogisticRegression(C=best_C)   # initialization and configuration
logclf.fit(X_train, y_train)    # training
print('\n')
print('The training score: {:.2f}%'
     .format(logclf.score(X_train, y_train)*100))
print('The validation score: {:.2f}%'
     .format(logclf.score(X_valid, y_valid)*100))
print('The test score: {:.2f}%'
     .format(logclf.score(X_test, y_test)*100))


# Decision boundary
print('\n')
print('Compressive Strength Logistic decision boundary \n ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(logclf.intercept_[0], logclf.coef_[0,0], logclf.coef_[0,1]))

# Model evaluation via the test set
x1_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
x1_plot = x1_plot.reshape(-1, 1)
x2_plot = -(logclf.coef_[0,0]*x1_plot + logclf.intercept_[0])/logclf.coef_[0,1]
plt.plot(x1_plot, x2_plot, '-', c='black', label='Logistic decision boundary')

y_predict = logclf.predict(X_test)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='>35 MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='<35 MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='>35 MPa')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='<35 MPa')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Compressive Strength Concrete test data vs Logistic prediction \n decision boundary ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(logclf.intercept_[0], logclf.coef_[0,0], logclf.coef_[0,1]), size=15)
plt.xlabel('glass feature Na (x1)', size=15)
plt.ylabel('glass feature Si (x2)', size=15)
plt.legend(loc='lower left', fontsize='medium')
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

# detailed evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_predict))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predict))
print('F1 score: %.3f' % f1_score(y_true=y_test, y_pred=y_predict))

###################################################################################################################################

# logistic classifier training with grid search
C_array = np.arange(0.01, 5.01, 0.01)
best_score = 0.0

from sklearn.svm import LinearSVC

for C_lsvc in C_array:
    clf = LinearSVC(C=C_lsvc, max_iter=1000000)
    clf.fit(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    if valid_score > best_score:
        best_score = valid_score
        best_C = C_lsvc

print('The best validation score: {:.2f}%'.format(best_score*100))
print('The best parameter C: {}'.format(best_C))

clf = LinearSVC(C=best_C, max_iter=1000000)   # initialization and configuration
clf.fit(X_train, y_train)    # training
print('\n')
print('The training score: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('The validation score: {:.2f}%'
     .format(clf.score(X_valid, y_valid)*100))
print('The test score: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))


# Decision boundary
print('\n')
print('Compressive Strength LSVC decision boundary \n ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(clf.intercept_[0], clf.coef_[0,0], clf.coef_[0,1]))

# Model evaluation via the test set
x1_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
x1_plot = x1_plot.reshape(-1, 1)
x2_plot = -(clf.coef_[0,0]*x1_plot + clf.intercept_[0])/clf.coef_[0,1]
plt.plot(x1_plot, x2_plot, '-', c='black', label='LSVC decision boundary')

y_predict = clf.predict(X_test)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='>35 MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='<35MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='>35MPa test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='<35MPa test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Compressive Strength test data vs LSVC prediction \n decision boundary ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(clf.intercept_[0], clf.coef_[0,0], clf.coef_[0,1]), size=15)
plt.xlabel('Concrete Feature FA (x1)', size=15)
plt.ylabel('Concret Feature CA (x2)', size=15)
plt.legend(loc='lower left', fontsize='medium')
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

# detailed evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_predict))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predict))
print('F1 score: %.3f' % f1_score(y_true=y_test, y_pred=y_predict))

#########################################################################################################################################

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

plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='>35 MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='<35MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='>35MPa test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='<35MPa test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Compressive Strength test data vs rbf SVC \n test score = {:.2f}% (gamma={:.2f} and C={:.2f}) '
          .format((clf.score(X_test, y_test)*100), best_parameters['gamma'], best_parameters['C']), size=15)
plt.xlabel('concrete feature FA (x1)', size=15)
plt.ylabel('concrete feature CA (x2)', size=15)
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

# detailed evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_predict))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predict))
print('F1 score: %.3f' % f1_score(y_true=y_test, y_pred=y_predict))

"""
Discussion:
The two features that I chose to try to predict the compressive strength classes were the amounts of fine aggregate and 
coarse aggregate. The logistic regression model was able to predict the correct class about 58% of the time. The model predicted the correct class at 
about the same rate for both positives and negatives. This particular model and test set resulted in 73 true negatives, 53 false negatives, 
33 false positives, and 47 true positives. The results from the LSVC model were very similar to the results of the logistic regression model.
The only difference between the two confusion matricies is that the LSVC confusion matrix predicted one more false positive, which shows the
logistic regression is the better model between the two. The rbf SVC model yielded the best results out of all three models. This modeled predicted
76 true negatives, 43 false negatives, 30 false positives, and 57 true positives. This model's rate for correct predictions is about 65%.
In conclusion, when using the FA and CA features for the prediction of compressive strength of concrete, the best model to use for the best results
is the rbf SVC model.

"""