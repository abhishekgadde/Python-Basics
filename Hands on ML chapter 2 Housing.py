#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
housing=pd.read_csv(r'C:\Users\Abhishek\Desktop\Hands on Machine Learning\housing.csv')
housing.head()


# In[9]:


housing.info()


# In[10]:


housing.describe()


# In[11]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
# bins the number of bars you have to show in your histogram plot.


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[ ]:


# Here we consider median_income as an attribute because it is most related to the target variable.


# In[14]:


import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]


# In[16]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[18]:


#Visualizing Geographical Data
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")


# In[19]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# Setting alpha = 0.1 makes it much easier to visulize the data


# In[20]:



housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[22]:


# Looking for Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[23]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[24]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)


# In[25]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[26]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


# Preparing dataset for machine learning algorithms


# In[27]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[ ]:


# Data Cleaning
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

