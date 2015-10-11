
# coding: utf-8

# In[26]:

import pandas as pd


# In[4]:

sales = pd.read_csv("home_data.csv")


# In[17]:

sales


# ## Selection and Summary Statistics

# In[92]:

expensive = sales[sales['zipcode'] == 98039]    # most expensive neighberhood
print expensive.price.mean()    # price average of the most expensive neighberhood


# ## Filtering Data

# In[102]:

filtered_data = sales[(sales['sqft_living'] > 2000)]
filtered_data = filtered_data[(filtered_data['sqft_living'] <= 4000)]
print filtered_data.shape[0]/float(sales.shape[0]) # Dimension of the filtered_data


# ## Building a regression model with several more features 

# In[82]:

# import sklearn libraries
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error
from math import sqrt


# In[137]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors 
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
# define features and predictor
y = sales['price']
x = sales[advanced_features]
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=3)


# ### Compute the RMSE

# In[138]:

# fitting the model with the advanced features
clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)
rmse1 = sqrt(mean_squared_error(y_test, clf.predict(x_test)))  # advanced_features
print rmse1


# In[139]:

# fitting the model with my_features
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
clf.fit(x_train[my_features],y_train)
rmse2 = sqrt(mean_squared_error(y_test, clf.predict(x_test[my_features]))) # my_features
print rmse2


# In[140]:

print rmse2 - rmse1

