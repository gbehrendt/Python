# ME:4150 Artificial Intelligence in Engineering
# Homework 8
# by Gabriel Behrendt

# import pandas library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")     #ignore warnings
np.set_printoptions(precision=2)    # keep 2 digits when printing a float array

#read from excel file
concrete = pd.read_excel("concrete.xls")

# two variables "tensile strength" as input and "endurance limit" as output
X = concrete['cement'].values
y = concrete['Cstr'].values

plt.scatter(X, y, marker = '+', c='k')
plt.xlabel('Cement', size=15)
plt.ylabel('Compressive Strength (MPa)', size=15)
plt.title('Test results', size= 15)
plt.show()

# training/validation/test sets splitting
X=X.reshape(-1,1)   # Samples are in column vector
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('\n')
print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))
print('\n')

# adding polynomial features 
from sklearn.preprocessing import PolynomialFeatures
ndegree = 12
poly = PolynomialFeatures(degree = ndegree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


# linear regression for polynomial
from sklearn.linear_model import LinearRegression
linreg_poly = LinearRegression().fit(X_train_scaled, y_train)


# Lasso regularization with cross-validation
a_array = np.arange(0.001, 0.01, 0.001)
best_score = 0.0

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
for a_lasso in a_array:
    linlasso = Lasso(alpha=a_lasso, max_iter = 10000000)
    valid_score = cross_val_score(linlasso, X_train_scaled, y_train, cv=5)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_a = a_lasso

print("The best alpha for lasso regularization is {}".format(best_a))

# retrain the best model
linlasso= Lasso(alpha=best_a, max_iter = 10000000)
linlasso.fit(X_train_scaled, y_train)    # training

print('\n')
print('Without regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linreg_poly.intercept_, linreg_poly.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(test) '
     .format(linreg_poly.score(X_train_scaled, y_train), linreg_poly.score(X_test_scaled, y_test)))

print('\n')
print('With lasso regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linlasso.intercept_, linlasso.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(test) '
     .format(linlasso.score(X_train_scaled, y_train), linlasso.score(X_test_scaled, y_test)))

# plot the ridge regression model 
X_plot=np.arange(X_train.min(), X_train.max(), 0.1)
X_plot.shape=(X_plot.size,1)
X_plot_poly = poly.fit_transform(X_plot)
X_plot_scaled = scaler.transform(X_plot_poly)
y_plot=linreg_poly.predict(X_plot_scaled)
y_plot_lasso=linlasso.predict(X_plot_scaled)

plt.figure()
plt.scatter(X_test, y_test, marker= 'o', c='red', s=50, label='test set')
plt.plot(X_plot, y_plot, 'g-', label='w/o regularization (n={})'.format(ndegree), linewidth =3)
plt.plot(X_plot, y_plot_lasso, 'b-', label='Lasso regularization', linewidth =3)
plt.legend(loc='upper left', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.title('Lasso Regularization ($\\alpha$ = {:.4f}) \n training score ({:.3f}) and test score ({:.3f})'
          .format(best_a, linlasso.score(X_train_scaled, y_train), 
                  linlasso.score(X_test_scaled, y_test)), size=15)
plt.xlabel('Cement (X) ', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.show()

