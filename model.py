# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:42:43 2022

@author: anjuc
"""
#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('merged.csv')

df=df.drop(['order_id', 'order_item_id', 
            'product_id','seller_id','seller_zip_code_prefix',
            'customer_id','customer_unique_id','customer_zip_code_prefix',
            'review_id'],axis=1)

df=df.drop(['review_comment_title',             
'review_comment_message',           
'review_creation_date',                 
'review_answer_timestamp'],axis=1)

df=df.drop_duplicates(keep='first')

Date=['shipping_limit_date', 'order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','order_approved_at']
for i in Date:
    df[i]=pd.to_datetime(df[i])
    
    
#droping null values in day time column
df=df.dropna(subset=['order_approved_at','order_delivered_carrier_date','order_delivered_customer_date'])
#df=df.dropna(subset='order_approved_at')
#df=df.dropna(subset='order_delivered_carrier_date')
#df=df.dropna(subset='order_delivered_customer_date')


# Using Lambda function to create a new target column
df['ratings']=df['review_score'].apply(lambda x:1 if x>3 else 0)
df=df.drop(['review_score'],axis=1)

#handling missing values
df['product_category_name']=df['product_category_name'].fillna(df['product_category_name'].mode()[0])
df['product_name_lenght']=df['product_name_lenght'].fillna(df['product_name_lenght'].median())
df['product_description_lenght']=df['product_description_lenght'].fillna(df['product_description_lenght'].median())
df['product_photos_qty']=df['product_photos_qty'].fillna(df['product_photos_qty'].median())
df['product_weight_g']=df['product_weight_g'].fillna(df['product_weight_g'].median())
df['product_length_cm']=df['product_length_cm'].fillna(df['product_length_cm'].median())
df['product_height_cm']=df['product_height_cm'].fillna(df['product_height_cm'].median())
df['product_width_cm']=df['product_width_cm'].fillna(df['product_width_cm'].median())


# Detecting and handling outliers in price 
df['price']=np.log(df['price']) 
q1,q2,q3 = np.percentile(df['price'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['price']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['price']=np.where(df['price']>upper_limit,upper_limit,df['price']) # Capping the upper limit
df['price']=np.where(df['price']<lower_limit,upper_limit,df['price']) # Flooring the lower limit

# Detecting and handling outliers in freight_value 
df['freight_value']=np.log(df['freight_value'])
q1,q2,q3 = np.percentile(df['freight_value'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['freight_value']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['freight_value']=np.where(df['freight_value']>upper_limit,upper_limit,df['freight_value'])
df['freight_value']=np.where(df['freight_value']<lower_limit,upper_limit,df['freight_value'])

# Detecting and handling outliers in product_name_length
q1,q2,q3 = np.percentile(df['product_name_lenght'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_name_lenght']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_name_lenght']=np.where(df['product_name_lenght']>upper_limit,upper_limit,df['product_name_lenght'])
df['product_name_lenght']=np.where(df['product_name_lenght']<lower_limit,upper_limit,df['product_name_lenght'])

# Detecting and handling outliers in  product_description_lenght 
q1,q2,q3 = np.percentile(df['product_description_lenght'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_description_lenght']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_description_lenght']=np.where(df['product_description_lenght']>upper_limit,upper_limit,df['product_description_lenght'])
df['product_description_lenght']=np.where(df['product_description_lenght']<lower_limit,upper_limit,df['product_description_lenght'])

# Detecting and handling outliers in product_photos_qty  
q1,q2,q3 = np.percentile(df['product_photos_qty'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_photos_qty']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_photos_qty']=np.where(df['product_photos_qty']>upper_limit,upper_limit,df['product_photos_qty'])
df['product_photos_qty']=np.where(df['product_photos_qty']<lower_limit,upper_limit,df['product_photos_qty'])

# Detecting and handling outliers in product_weight_g   
q1,q2,q3 = np.percentile(df['product_weight_g'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_weight_g']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_weight_g']=np.where(df['product_weight_g']>upper_limit,upper_limit,df['product_weight_g'])
df['product_weight_g']=np.where(df['product_weight_g']<lower_limit,upper_limit,df['product_weight_g'])        
        
# Detecting and handling outliers in product_length_cm   
q1,q2,q3 = np.percentile(df['product_length_cm'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_length_cm']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_length_cm']=np.where(df['product_length_cm']>upper_limit,upper_limit,df['product_length_cm'])
df['product_length_cm']=np.where(df['product_length_cm']<lower_limit,upper_limit,df['product_length_cm'])       
        
# Detecting and handling outliers in product_height_cm    
q1,q2,q3 = np.percentile(df['product_height_cm'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_height_cm']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_height_cm']=np.where(df['product_height_cm']>upper_limit,upper_limit,df['product_height_cm'])
df['product_height_cm']=np.where(df['product_height_cm']<lower_limit,upper_limit,df['product_height_cm'])    

 # Detecting and handling outliers inproduct_width_cm   
q1,q2,q3 = np.percentile(df['product_width_cm'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['product_width_cm']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['product_width_cm']=np.where(df['product_width_cm']>upper_limit,upper_limit,df['product_width_cm'])
df['product_width_cm']=np.where(df['product_width_cm']<lower_limit,upper_limit,df['product_width_cm'])
        
 # Detecting and handling outliers in payment_installments'  
q1,q2,q3 = np.percentile(df['payment_installments'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['payment_installments']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
df['payment_installments']=np.where(df['payment_installments']>upper_limit,upper_limit,df['payment_installments'])
df['payment_installments']=np.where(df['payment_installments']<lower_limit,upper_limit,df['payment_installments'])     
        
# Detecting and handling outliers in payment_value'  
q1,q2,q3 = np.percentile(df['payment_value'],[25,50,75])
IQR = q3-q1
lower_limit = q1-(1.5*IQR)
upper_limit = q3+(1.5*IQR)
outlier = []
for i in df['payment_value']:
    if ((i>upper_limit) or (i<lower_limit)):
        outlier.append(i)
        
df['payment_value']=np.where(df['payment_value']>upper_limit,upper_limit,df['payment_value'])
df['payment_value']=np.where(df['payment_value']<lower_limit,upper_limit,df['payment_value'])      

#creating new columns
#Time of estimated delivery
df['estimated_time'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).apply(lambda x: x.total_seconds()/3600)
df['actual_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).apply(lambda x: x.total_seconds()/3600)
df['diff_actual_estimated'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).apply(lambda x: x.total_seconds()/3600)
df['diff_approval_shipping'] = (df['shipping_limit_date'] - df['order_approved_at']).apply(lambda x: x.total_seconds()/3600)
df['shipping_time'] = (df['order_delivered_carrier_date'] - df['shipping_limit_date']).apply(lambda x: x.total_seconds()/3600)
df['product_size'] = (df['product_length_cm']*df['product_width_cm']*df['product_height_cm'])
#dropping columns
df=df.drop(['order_delivered_customer_date'],axis=1)
df=df.drop(['order_purchase_timestamp'],axis=1)
df=df.drop(['order_estimated_delivery_date'],axis=1)
df=df.drop(['shipping_limit_date'],axis=1)
df=df.drop(['order_approved_at'],axis=1)
df=df.drop(['order_delivered_carrier_date'],axis=1)
df=df.drop(['payment_sequential'],axis=1)
df=df.drop(['product_length_cm'],axis=1)
df=df.drop(['product_width_cm'],axis=1)
df=df.drop(['product_height_cm'],axis=1)


#encoding
#categorical_features=[feature for feature in df.columns if df[feature].dtype =='O']

#Label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

#for column in categorical_features:
    #df[column] =label_encoder.fit_transform(df[column])
    #filename= 'encoding.pkl'
    #pickle.dump(label_encoder,open(filename,"wb"))

product_category_name=label_encoder.fit_transform(df['product_category_name'])
df['product_category_name']=product_category_name
pickle.dump(label_encoder,open('product_category_name.pkl','wb'))
print(product_category_name)


seller_city=label_encoder.fit_transform(df['seller_city'])
df['seller_city']=seller_city
pickle.dump(label_encoder,open('seller_city.pkl','wb'))
print(seller_city)

seller_state=label_encoder.fit_transform(df['seller_state'])
df['seller_state']=seller_state
pickle.dump(label_encoder,open('seller_state.pkl','wb'))
print(seller_state)

order_status=label_encoder.fit_transform(df['order_status'])
df['order_status']=order_status
pickle.dump(label_encoder,open('order_status.pkl','wb'))
print(order_status)

payment_type=label_encoder.fit_transform(df['payment_type'])
df['payment_type']=payment_type
pickle.dump(label_encoder,open('payment_type.pkl','wb'))
print(payment_type)

customer_city=label_encoder.fit_transform(df['customer_city'])
df['customer_city']=customer_city
pickle.dump(label_encoder,open('customer_city.pkl','wb'))
print(customer_city)


customer_state=label_encoder.fit_transform(df['customer_state'])
df['customer_state']=customer_state
pickle.dump(label_encoder,open('customer_state.pkl','wb'))
print(customer_state)

#splitting data into dependent and independent columns
x=df.drop('ratings',axis=1)
y=df['ratings']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.33)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model=rf.fit(x_train,y_train)
pickle.dump(rf,open('model.pkl','wb'))
