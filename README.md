
# Ridge and Lasso Regression

## Introduction

At this point, you've seen a number of criteria and algorithms for fitting regression models to data. You've seen the simple linear regression using ordinary least squares, and its more general regression of polynomial functions. You've also seen how we can overfit models to data using polynomials and interactions. With all of that, you began to explore other tools to analyze this general problem of overfitting versus underfitting, all this using train and test splits, bias and variance, and cross validation.

Now you're going to take a look at another way to tune the models you create. These methods all modify the mean squared error function that you are optimizing against. The modifications will add a penalty for large coefficient weights in the resulting model.

## Objectives

You will be able to:

- Understand what Lasso regression is 
- Understand what Ridge regression is
- Compare and contrast Lasso and Ridge regression 
- Understand that both Lasso and Ridge regression can be used to counter multicollinearity

## Our regression cost function

From previously, you know that when solving for a linear regression, you can express the cost function as

$$ \text{cost_function}= \sum_{i=1}^n(y_i - \hat{y})^2 = \sum_{i=1}^n(y_i - (mx_i + b))^2$$

This is the expression for simple linear regression (for 1 predictor $x$). If you have multiple predictors, you would have something that looks like:

$$ \text{cost_function}= \sum_{i=1}^n(y_i - \hat{y})^2 = \sum_{i=1}^n(y_i - \sum_{j=1}^k(m_jx_{ij} ) -b )^2$$

where $k$ is the number of predictors.

## Penalized estimation

You've seen that when the number of predictors increases, your model complexity increases, with a higher chance of overfitting as a result. We've previously seen fairly ad-hoc variable selection methods (such as forward/backward selection), to simply select a few variables from a longer list of variables as predictors. 

Now, instead of completely "deleting" certain predictors from a model (which is equal to setting coefficients equal to zero), wouldn't it be interesting to just reduce the values of the coefficients to make them less sensitive to noise in the data? *Penalized estimation* operates in a way where parameter shrinkage effects are used to make some or 
all of the coefficients smaller in magnitude (closer to zero). Some of the penalties have the property of performing both variable selection (setting some coefficients exactly equal to zero) and shrinking the other coefficients. Ridge and Lasso regression are two examples of penalized estimation. There are multiple advantages to using these methods:

- They reduce model complexity
- The may prevent from overfitting
- Some of them may perform variable selection at the same time (when coefficients are set to 0)
- They can be used to counter multicollinearity

Lasso and Ridge are two commonly used so-called **regularization techniques**. Regularization is a general term used when one tries to battle overfitting. Regularization techniques will be covered in more depth when we're moving into machine learning!

## Ridge regression

In ridge regression, the linear regression cost function is changed by adding a penalty term to the square of the magnitude of the coefficients.

$$ \text{cost_function_ridge}= \sum_{i=1}^n(y_i - \hat{y})^2 = \sum_{i=1}^n(y_i - \sum_{j=1}^k(m_jx_{ij})-b)^2 + \lambda \sum_{j=1}^p m_j^2$$

If you have two predictors the full equation would look like this (Notice that there is a penalty term `m` for each predictor in the model - in this case, two) : 

$$ \text{cost_function_ridge}= \sum_{i=1}^n(y_i - \hat{y})^2 = $$

$$ \sum_{i=1}^n(y_i - ((m_1x_{1i})-b)^2 + \lambda m_1^2 + (m_2x_{2i})-b)^2 + \lambda m_2^2)$$

Recall that you want to minimize your cost function, so by adding the penalty term $\lambda$, ridge regression puts a constraint on the coefficients $m$. This means that large coefficients penalize the optimization function. That's why ridge regression leads to a shrinkage of the coefficients and helps to reduce model complexity and multicollinearity.


$\lambda$ is a so-called *hyperparameter*, which means you have to specify the value for lambda. For a small lambda, the outcome of your ridge regression will resemble a linear regression model. For large lambda, penalization will increase and more parameters will shrink.

Ridge regression is often also referred to as **L2 Norm Regularization**


## Lasso regression

Lasso regression is very similar to Ridge regression, except that the magnitude of the coefficients are not squared in the penalty term. So, while ridge regression keeps the sum of the squared regression coefficients (except for the intercept) bounded, the lasso method bounds the sum of the absolute values.

The resulting cost function looks like this:

$$ \text{cost_function_lasso}= \sum_{i=1}^n(y_i - \hat{y})^2 = \sum_{i=1}^n(y_i - \sum_{j=1}^k(m_jx_{ij})-b)^2 + \lambda \sum_{j=1}^p \mid m_j \mid$$

If you have two predictors the full equation would look like this (Notice that there is a penalty term (m) for each predictor in the model - in this case, two) : 

$$ \text{cost_function_lasso}= \sum_{i=1}^n(y_i - \hat{y})^2 = $$

$$\sum_{i=1}^n(y_i - ((m_1x_{1i})-b)^2 + \lambda \mid m_1 \mid) + ((m_2x_{2i})-b)^2 + \lambda \mid m_2 \mid) $$

The name "Lasso" comes from ‘Least Absolute Shrinkage and Selection Operator’.

While looking similar to the definition of the ridge estimator, the effect of the absolute values is that some coefficients might be set exactly equal to zero, while other coefficients are shrunk towards zero. Hence the lasso method is attractive because it performs estimation *and* selection simultaneously. Especially for variable selection when the number of predictors is very high.

Lasso regression is often also referred to as **L1 Norm Regularization**


### Standardization before Regularization

An important step before using either Lasso or Ridge regularization is to first standardize your data such that it is all on the same scale. Regularization is based on the concept of penalizing larger coefficients, so if you have features that are on different scales, some will get unfairly penalized. Below, you can see that we are using a MinMaxScaler to standardize our data to the same scale. A downside of standardization is that the value of the coefficients become less interpretable and must be transformed back to their original scale if you want to interpret how a one unit change in a feature impacts the target variable.

## An example using our `auto-mpg` data

Let's transform our continuous predictors in `auto-mpg` and see how they perform as predictors in a Ridge versus Lasso regression.


```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("auto-mpg.csv") 
data['horsepower'].astype(str).astype(int)
y = data[["mpg"]]
X = data.drop(["mpg", "car name", "origin"], axis=1)

scale = MinMaxScaler()
transformed = scale.fit_transform(X)
X = pd.DataFrame(transformed, columns = X.columns)
```

Below, we've performed train-test-splits and fit Ridge, Lasso and Linear regression models. Notice that the Ridge and Lasso models have the parameter alpha, which is Scikit's version of $\lambda$ in the regularization cost functions.


```python
# Perform test train split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Build a Ridge, Lasso and regular linear regression model. 
# Note how in scikit learn, the regularization parameter is denoted by alpha (and not lambda)
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)

lin = LinearRegression()
lin.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Next, let's create predictions for train and test sets.


```python
# Create preditions for training and test sets
y_h_ridge_train = ridge.predict(X_train)
y_h_ridge_test = ridge.predict(X_test)

y_h_lasso_train = np.reshape(lasso.predict(X_train), (274,1))
y_h_lasso_test = np.reshape(lasso.predict(X_test), (118,1))

y_h_lin_train = lin.predict(X_train)
y_h_lin_test = lin.predict(X_test)
```

Look at the RSS for train and test for each of the three models.


```python
print('Train Error Ridge Model', np.sum((y_train - y_h_ridge_train)**2))
print('Test Error Ridge Model', np.sum((y_test - y_h_ridge_test)**2))
print('\n')

print('Train Error Lasso Model', np.sum((y_train - y_h_lasso_train)**2))
print('Test Error Lasso Model', np.sum((y_test - y_h_lasso_test)**2))
print('\n')

print('Train Error Unpenalized Linear Model', np.sum((y_train - lin.predict(X_train))**2))
print('Test Error Unpenalized Linear Model', np.sum((y_test - lin.predict(X_test))**2))
```

    Train Error Ridge Model mpg    2688.222824
    dtype: float64
    Test Error Ridge Model mpg    2074.197775
    dtype: float64
    
    
    Train Error Lasso Model mpg    4644.536425
    dtype: float64
    Test Error Lasso Model mpg    3696.183375
    dtype: float64
    
    
    Train Error Unpenalized Linear Model mpg    2658.043444
    dtype: float64
    Test Error Unpenalized Linear Model mpg    1976.266987
    dtype: float64


We note that Ridge is clearly better than Lasso here, but that the unpenalized model performs best here. Let's see how including Ridge and Lasso changed our parameter estimates.


```python
print('Ridge parameter coefficients:', ridge.coef_)
print('Lasso parameter coefficients:', lasso.coef_)
print('Linear model parameter coefficients:', lin.coef_)
```

    Ridge parameter coefficients: [[ -2.11792413  -3.0112953   -1.90579654 -15.60758962  -1.61071692
        8.12940111]]
    Lasso parameter coefficients: [-10.31005725  -0.          -0.          -2.27967948   0.
       3.88327477]
    Linear model parameter coefficients: [[ -1.33790698  -1.05300843  -0.08661412 -20.08143923  -0.39639115
        8.56051229]]


You can clearly see how Lasso shrinks certain parameters to 0! The Ridge regression mostly affected the fourth parameter (estimated to be -20.08 for the linear regression model).

## Additional reading

Full code examples for Ridge and Lasso regression, advantages and disadvantages, and how to code ridge and Lasso in Python can be found [here](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/).

Make sure to have a look at the Scikit-Learn documentation for [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).


## Summary

Great! You now know how to perform Lasso and Ridge regression. Let's move on to the lab to explore Lasso and Ridge further!
