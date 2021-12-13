#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta, datetime
from scipy.sparse import coo_matrix, hstack, csr_matrix


# In[2]:


print("="*100)
print("Importing Sample Data")
directory = 'C:\\Users\\Emma Hegermiller\\Git\\price-prediction\\data'

df_import =pd.read_csv(directory + '\\sample.csv')
print("Import Complete")
print("="*100)


# In[3]:


# Drop invalid row that are missing vin or list_price
# Drop variables that are not vehicle attributes
# Drop variables that cannot be used in price predication at the time of listing
print("="*100)
print("Preprocessing Data")
drop_cols = ['index_pandas', 
             'days_on_market', 
             'listing_date_end', 'shift_buyer_region']
df_valid = df_import.dropna(subset=['vin', 'list_price']).drop(columns = drop_cols).drop_duplicates().astype({'year': 'str'})
print("Columns: n/{}".format(df_valid.columns))
print("Row count: {}".format(len(df_valid)))
df_valid.sort_values(["vin", "listing_date_begin"]).head(100)
print("="*100)


# In[4]:


# Missingness
print("="*100)
print("Evaluate missingness")
row_total = len(df_valid)
print(df_valid.isna().sum()/row_total)
print("="*100)


# In[5]:


# Calculate data values for imputation
print("="*100)
print("Values for imputation")

imputation_dict = {}

# Get median values for imputation of numeric variables
numeric_cols = ['mileage', 'accident_count',
       'fuel_economy_city', 'fuel_economy_highway', 'msrp']
for col in numeric_cols:
    median = df_valid[col].median()
    imputation_dict[col] = median
    
# Get imputation values for boolean variables
boolean_cols = ['is_cpo', 'seller_is_franchise_dealer', 'seller_is_online_only', 'seller_ships_nationwide']
for col in boolean_cols:
    if len(df_valid[df_valid[col]==True])==0:
        imputation_dict[col] = True
    if len(df_valid[df_valid[col]==False])==0:
        imputation_dict[col] = False
    else:
        pass

# Get imputation for categorical variables with majority categies with subcategory counts < 0.1%
text_cols = ["trim", "transmission", "exterior_color"]
for col in text_cols:
    imputation_dict[col] = 'Other'

# Get imputation for other categorical variables 
categorical_cols = ['year', 'make', 'model', 'body_style', 'seller_city', 'seller_state', 'seller_type']
for col in categorical_cols:
    imputation_dict[col] = df_valid[col].mode()[0]
imputation_dict['seller_city'] = df_valid['seller_city'].mode()[0].split(',')[0]

print(imputation_dict)
print("="*100)  


# In[6]:


print("="*100)
print("Split into test and ")
# Split into train and test
unique_vins = df_valid['vin'].unique()
print("Number of unique vins: {}".format(len(unique_vins)))

train_vin = np.random.choice(unique_vins, size=int(len(unique_vins)*0.8), replace=False, p=None)
test_vin = unique_vins[~np.in1d(unique_vins, train_vin)]

df_train = df_valid[df_valid['vin'].isin(train_vin)]
df_test = df_valid[df_valid['vin'].isin(test_vin)]
print("test row percentage: {}%".format(round(100*(len(df_test)/len(df_valid)))))

# Check takes long to run
# vin_duplicated = []
# print("Checking uniqueness of train and test vins")
# for i in df_test.index:
#     if df_test.loc[i, 'vin'] in list(df_train['vin']):
#         vin_duplicated.append(df_test.loc[i, 'vin'])

# assert len(vin_duplicated)==0, "Vin number in test and train"
print("="*100)


# In[7]:


# Group by observations to summarize 
print("="*100)
print("Group by observations to summarize")

def summarize_vin(group1):
    """Summarizes observations using mode for categorical variables and median for numeric variables"""
    if len(group1) > 1:
        for col in group1.columns:
            if group1[col].count() > 1:
                if np.issubdtype(group1[col], np.number):
                    group1[col] = group1[col].median()
                else:
                    group1[col] = group1[col].value_counts().index[0]
    return group1

def groupby_vehicle(data):
    """Groups by vin and summarizes categorical and numberic variables that are considered static vehicle attributes"""
    static_cols = ['vin',
                   'year', 
                   'make', 
                   'model', 
                   'trim', 
                   'body_style', 
                   'transmission', 
                   'fuel_economy_city', 
                   'fuel_economy_highway', 
                   'exterior_color']
    df_out = data.groupby('vin')[static_cols].apply(summarize_vin).drop_duplicates()
    return df_out

def groupby_listing(data):
    """Groups by vin and listing_date_begin. 
    Summarizes categorical and numberic variables that are considered static vehicle attributes"""
    dynamic_cols = ['vin', 
                'listing_date_begin',
                'mileage', 
                'accident_count',
                'is_cpo',
                'seller_city', 
                'seller_state', 
                'seller_type',
                'seller_is_franchise_dealer', 
                'seller_is_online_only',
                'seller_ships_nationwide', 
                'msrp',
                'list_price']
    df_out = data.groupby(['vin', 'listing_date_begin'])[dynamic_cols].apply(summarize_vin).drop_duplicates()
    return df_out
    
def summarize_obs(data):
    df_vehicle = groupby_vehicle(data)
    df_listing = groupby_listing(data)
    df_out = df_listing.join(df_vehicle.set_index("vin"), on="vin")
    return df_out
    
df_tidy = summarize_obs(df_train)

print("Columns: n/{}".format(df_tidy.columns))
print("Length: {}".format(len(df_tidy)))
print(df_tidy.sort_values(["vin", "listing_date_begin"]).head(100))

# Test
df_tidy_test = summarize_obs(df_test)

print("="*100)


# In[8]:


# Impute missing values
print("="*100)
print("Impute missing values")

def impute_variables(data):
    """
    Use dataset values for imputation of missing variables
    """
    return data.fillna(value=imputation_dict).replace({True:1, False:0})

df_impute = impute_variables(df_tidy)

print("Columns: n/{}".format(df_impute.columns))
print("Length: {}".format(len(df_impute)))
print(df_impute.sort_values(["vin", "listing_date_begin"]).head(100))

# Test
df_impute_test = impute_variables(df_tidy_test)
print("="*100)


# In[9]:


# Date Handling
print("="*100)
print("Handle date variable")

def handle_date(data):
    #Extracting Yea
    data['listing_year'] = pd.to_datetime(data['listing_date_begin']).dt.year
    #Extracting Month
    map_month = {1:'Jan', 2:'Feb', 3:'Mar', 
                 4:'Apr', 5:'May', 6:'Jun', 
                 7:'Jul', 8:'Aug', 9:'Sep', 
                 10:'Oct', 11:'Nov', 12:'Dec'}
    data['listing_month'] = pd.to_datetime(data['listing_date_begin']).dt.month.replace(map_month)
    return data

df_date = handle_date(df_impute)

print("Columns: n/{}".format(df_date.columns))
print("Length: {}".format(len(df_date)))
print(df_date.sort_values(["vin", "listing_date_begin"]).head(100))

# Test
df_date_test = handle_date(df_impute_test)
print("="*100)


# In[10]:


# Split features with where it makes sense
print("="*100)
print("Splitting seller city")
df_date['seller_city'] = df_date.seller_city.str.split(",").str[0]

# Test
df_date_test['seller_city'] = df_date_test.seller_city.str.split(",").str[0]
print("="*100)


# In[11]:


# Handling Outliers
print("="*100)
print("Handle Outliers")

def handle_outliers(data):
    numeric_cols = ['mileage', 'accident_count',
       'fuel_economy_city', 'fuel_economy_highway', 'msrp']
    print("Input Summary")
    print(data.describe())
    for col in numeric_cols:
        upper_lim = data[col].quantile(.95)
        lower_lim = data[col].quantile(.05)
        data_out = data[(data[col] < upper_lim) & (data[col] > lower_lim)]
    return data_out

df_outlier = handle_outliers(df_date)

print("Columns: n/{}".format(df_outlier.columns))
print("Length: {}".format(len(df_outlier)))
print("Output Summary")
print(df_outlier.describe())
print(df_outlier.sort_values(["vin", "listing_date_begin"]).head(100))

# Test
df_outlier_test = handle_outliers(df_date_test)
print("="*100)


# In[12]:


# Normalization
print("="*100)
print("Normalization")

def normalize(data):
    numeric_cols = ['mileage', 'accident_count','fuel_economy_city', 'fuel_economy_highway', 'msrp']
    for col in numeric_cols:
        data[col+'_normalized'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

df_norm = normalize(df_outlier)

print("Columns: n/{}".format(df_norm.columns))
print("Length: {}".format(len(df_norm)))
print(df_norm.describe())
print(df_norm.sort_values(["vin", "listing_date_begin"]).head(100))

# Test
df_norm_test = normalize(df_outlier_test)
print("="*100)


# In[13]:


# One hot encoding
# https://towardsdatascience.com/pricing-objects-in-mercari-machine-learning-deep-learning-perspectives-6aa80514b2c8
print("="*100)
print("One hot encoding")

def one_hot_encode(train,test,option):
    '''
    Function to one hot encode the categorical columns.
    train: train data used for fitting the CountVecotrizer and transformation
    test: test data used for in transformation
    option: 'test' or 'train' for resulting output
    '''
    vectorizer = OneHotEncoder(handle_unknown='ignore')
    vectorizer_total = vectorizer.fit(train[['year', 'make', 'model', 'body_style', 'seller_city',
                                            'seller_state', 'seller_type', 
                                            'listing_year', 'listing_month']])
#     vectorizer_year = vectorizer.fit(train['year'].astype('str').to_numpy().reshape(-1, 1)) 
#     vectorizer_make = vectorizer.fit(train['make'].values)
#     vectorizer_model = vectorizer.fit(train['model'].values)
#     vectorizer_body_style = vectorizer.fit(train['body_style'].values)
#     vectorizer_seller_city = vectorizer.fit(train['seller_city'].astype(str).values) 
#     vectorizer_seller_state = vectorizer.fit(train['seller_state'].astype(str).values)
#     vectorizer_seller_type = vectorizer.fit(train['seller_type'].astype(str).values)
#     vectorizer_listing_year = vectorizer.fit(train['listing_year'].astype(str).values) 
#     vectorizer_listing_month = vectorizer.fit(train['listing_month'].astype(str).values) 

    if option == 'test':
        print("{} one hot encoded column shapes".format(option))
        return_matrix = vectorizer.transform(test[['year', 'make', 'model', 'body_style', 'seller_city',
                                              'seller_state', 'seller_type', 
                                            'listing_year', 'listing_month']])
#         # vectorizing the 'year' column
#         column_year = vectorizer.transform(test['year'].astype('str').to_numpy().reshape(-1, 1)) 
#         print(column_year.shape)
#         #vectorizing the 'make' column
#         column_make = vectorizer.transform(test['make'].values)
#         print(column_make.shape)
#         #vectorizing 'model' column
#         column_model = vectorizer.transform(test['model'].values)
#         print(column_model.shape)
#         #vectorizing 'body_style' column
#         column_body_style = vectorizer.transform(test['body_style'].values)
#         print(column_body_style.shape)
#         #vectorizing 'seller_city' column
#         colummn_seller_city = vectorizer.transform(test['seller_city'].values)
#         print(colummn_seller_city.shape)
#         #vectorizing 'seller_state' column
#         colummn_seller_state = vectorizer.transform(test['seller_state'].values)
#         print(colummn_seller_state.shape)
#         #vectorizing 'seller_type' column
#         colummn_seller_type = vectorizer.transform(test['seller_type'].values)
#         print(colummn_seller_type.shape)
#         #vectorizing 'listing_year' column
#         colummn_listing_year = vectorizer.transform(test['listing_year'].values)  
#         print(colummn_listing_year.shape)
#         #vectorizing 'listing_month' column
#         colummn_listing_month = vectorizer.transform(test['listing_month'].values)
#         print(colummn_listing_month.shape)
    elif option == 'train':
        print("{} one hot encoded column shapes".format(option))
        return_matrix = vectorizer.transform(train[['year', 'make', 'model', 'body_style', 'seller_state', 'seller_type', 
                                            'seller_city', 'listing_year', 'listing_month']])
#         # vectorizing the 'year' column
#         column_year = vectorizer.transform(train['year'].astype('str').to_numpy().reshape(-1, 1)) 
#         print(column_year.shape)
#         #vectorizing the 'make' column
#         column_make = vectorizer.transform(train['make'].values)
#         print(column_make.shape)
#         #vectorizing 'model' column
#         column_model = vectorizer.transform(train['model'].values)
#         print(column_model.shape)
#         #vectorizing 'body_style' column
#         column_body_style = vectorizer.transform(train['body_style'].values)
#         print(column_body_style.shape)
#         #vectorizing 'seller_city' column
#         colummn_seller_city = vectorizer.transform(train['seller_city'].values)
#         print(colummn_seller_city.shape)
#         #vectorizing 'seller_state' column
#         colummn_seller_state = vectorizer.transform(train['seller_state'].values)
#         print(colummn_seller_state.shape)
#         #vectorizing 'seller_type' column
#         colummn_seller_type = vectorizer.transform(train['seller_type'].values)
#         print(colummn_seller_type.shape)
#         #vectorizing 'listing_year' column
#         colummn_listing_year = vectorizer.transform(train['listing_year'].values)  
#         print(colummn_listing_year.shape)
#         #vectorizing 'listing_month' column
#         colummn_listing_month = vectorizer.transform(train['listing_month'].values)
#         print(colummn_listing_month.shape)
    else:
        print("Need input option to specify return of test or train one hot encoded columns")
    
#     return_matrix = hstack((column_year,
# #                             column_make,column_model,column_body_style,
# #                          colummn_seller_city,colummn_seller_state,colummn_seller_type,
# #                          colummn_listing_year,colummn_listing_month
#                              )).tocsr()
    print("{} one hot encoded matrix shape".format(option))
    print(return_matrix.shape)
    return return_matrix



onehot = one_hot_encode(df_norm, df_norm_test, option='train')
onehot_test = one_hot_encode(df_norm, df_norm_test, option='test')
print("="*100)


# In[14]:


# Make, model, body_style price


# In[15]:


# Text feature extraction
# # vectorizing trim
# vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=250000)

# train_trim_tfidf = vectorizer.fit_transform(['trim'])
# # test_trim_tfidf = vectorizer.transform(test['trim'])


# # vectorixing transmission
# vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=500000)

# train_transmission_tfidf = vectorizer.fit_transform(train_df['transmission'])
# # test_transmission_tfidf = vectorizer.transform(test['transmission'])

# # vectorixing exterior_color
# vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=500000)

# train_exterior_color_tfidf = vectorizer.fit_transform(train_df['exterior_color'])
# # test_exterior_color_tfidf = vectorizer.transform(test['exterior_color'])


# In[16]:


# Confirm no missing values 
print("="*100)
print("Evaluating missingness")
# test_assert = pd.DataFrame([[np.nan, 1], [2, 5]], columns=["test_col1", "test_col2"])

# Train
row_total = len(df_norm)
na_cols = {}
for col in df_norm.columns:
    if df_norm[col].isna().sum()/row_total > 0:
        na_cols[col]=df_norm[col].isna().sum()/row_total
    else:
        pass

assert len(na_cols)==0, "Expecting there to be no missingness in train \n'column': missing\n {}".format(na_cols)    

# Test
row_total_test = len(df_norm_test)
test_na_cols = {}
for col in df_norm_test.columns:
    if df_norm_test[col].isna().sum()/row_total_test > 0:
        na_cols[col]=df_norm[col].isna().sum()/row_total
        test_na_cols[col] = df_norm_test[col].isna().sum()/row_total_test
    else:
        pass
    
assert len(test_na_cols)==0, "Expecting there to be no missingness in test \n'column': missing\n {}".format(test_na_cols) 
print("Evaluation complete")
print("="*100)


# In[17]:


# Confirm no categorical variables
print("="*100)
print("Check datatypes")
# Train
pd.set_option('display.max_rows', 500)
print("Train \n {}".format(df_norm.dtypes))

# Test
print("Test \n {}".format(df_norm_test.dtypes))
print("="*100)


# In[18]:


print("="*100)
print("Stacking train and test features")
X_train = hstack((df_norm.drop(columns=["vin","listing_date_begin", "trim", "transmission", "exterior_color", 
                                       "list_price", "mileage", "accident_count", 
                                        "fuel_economy_city", "fuel_economy_highway", "msrp", "year", "make",
                                        "model", "body_style", "seller_city", "seller_state", "seller_type", 
                                        "listing_year", "listing_month"]).to_numpy(), onehot))
print("Train feature matrix shape: \n {}".format(X_train.shape))
pd.DataFrame(data=csr_matrix.todense(X_train)).to_csv('{}\\X_train.csv'.format(directory), header=False)

X_test = hstack((df_norm_test.drop(columns=["vin","listing_date_begin", "trim", "transmission", "exterior_color", 
                                       "list_price", "mileage", "accident_count", 
                                        "fuel_economy_city", "fuel_economy_highway", "msrp", "year", "make",
                                        "model", "body_style", "seller_city", "seller_state", "seller_type", 
                                        "listing_year", "listing_month"]).to_numpy(), onehot_test))
print("Test feature matrix shape: \n {}".format(X_test.shape))
pd.DataFrame(data=csr_matrix.todense(X_test)).to_csv('{}\\X_test.csv'.format(directory), header=False)
print("="*100)


# In[19]:


print("="*100)
print("Train and test outcome")
y_train = df_norm.list_price.to_numpy()
pd.DataFrame(data=y_train).to_csv('{}\\y_train.csv'.format(directory), header=False)
print("Train outcome matrix shape: \n {}".format(y_train.shape))
y_test = df_norm_test.list_price.to_numpy()
pd.DataFrame(data=y_test).to_csv('{}\\y_test.csv'.format(directory), header=False)
print("Test outcome matrix shape: \n {}".format(y_test.shape))
print("="*100)


# In[20]:


import pandas as pd
from sklearn.metrics import mean_squared_error as mse # rsme = mean_squared_error(y_true, y_pred, squared=False)
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV as rcv
import sklearn.preprocessing as pp
import numpy as np
from joblib import parallel_backend


# In[33]:


# Ridge Regression Train
print("="*100)
print("Bulding Model")
with parallel_backend('threading', n_jobs=8):
    ridge = rcv(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0),cv=5).fit(X_train, y_train)
    ridge_params = ridge.get_params()
    r_squared = ridge.score(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    rmse_train = mse(y_train, y_train_pred, squared = False)


# In[34]:


# Ridge Regression Test
print("Testing Model")
with parallel_backend('threading', n_jobs=8):
    y_test_pred = ridge.predict(X_test)
    rmse_test = mse(y_test, y_test_pred, squared = False)


# In[36]:


# Model results
print("Model Results")
print("Ridge Regression Parameters: {}".format(ridge_params))
print("Ridge Regression alpha: {}".format(ridge.alpha_))
print("Ridge Train R^2: {}".format(r_squared))
print("Ridge Train RMSE: {}".format(rmse_train))
print("Ridge Test RMSE: {}".format(rmse_test))
print("="*100)

