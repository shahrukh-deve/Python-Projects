# In[1]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


# In[2]:


#Reading Data Set from csv file
data_set= pd.read_csv("carData.csv")


# In[3]:


#data_set2= pd.DataFrame(data_set)


# In[33]:


b = sns.countplot('manufacture_year', hue='manufacture_year', data=data_set.head(7))
b.set_xlabel("Manufacture year", fontsize=30)
b.set_ylabel('Price',fontsize=30)
plt.show()


# In[5]:


#Replacing nan values to Not specified and 0.0
data_set['maker']= data_set.maker.replace(np.nan, 'not specified')
data_set['model']= data_set.model.replace(np.nan, 'not specified')
data_set['transmission']= data_set.transmission.replace(np.nan, 'not specified')
data_set['fuel_type']= data_set.fuel_type.replace(np.nan, 'not specified')

data_set['mileage']= data_set.mileage.replace(np.nan, 0.0)
data_set['manufacture_year']= data_set.manufacture_year.replace(np.nan, 0.0)
data_set['engine_hp']= data_set.engine_hp.replace(np.nan, 0.0)
data_set['door_count']= data_set.door_count.replace(np.nan, 0.0)
data_set['seat_count']= data_set.seat_count.replace(np.nan, 0.0)

data_set['mileage']= data_set.mileage.replace('None', 0.0)
data_set['manufacture_year']= data_set.manufacture_year.replace('None', 0.0)
data_set['engine_hp']= data_set.engine_hp.replace('None', 0.0)
data_set['door_count']= data_set.door_count.replace('None', 0.0)
data_set['seat_count']= data_set.seat_count.replace('None', 0.0)


# In[6]:


data_set.head()


# In[7]:


maker_enc = LabelEncoder().fit(data_set['maker'])
data_set['maker']= maker_enc.transform(data_set['maker'])


# In[8]:


model_enc = LabelEncoder().fit(data_set['model'])
data_set['model']= model_enc.transform(data_set['model'])


# In[9]:


transmission_enc = LabelEncoder().fit(data_set['transmission'])
data_set['transmission']= transmission_enc.transform(data_set['transmission'])


# In[10]:


fuel_type_enc = LabelEncoder().fit(data_set['fuel_type'])
data_set['fuel_type']= fuel_type_enc.transform(data_set['fuel_type'])


# In[11]:


#engine_hp_enc = LabelEncoder().fit(data_set['engine_hp'])
#data_set['engine_hp']= engine_hp_enc.transform(data_set['engine_hp'])


# In[12]:


#mileage_enc = LabelEncoder().fit(data_set['mileage'])
#data_set['mileage']= mileage_enc.transform(data_set['mileage'])


# In[13]:


#door_count_enc = LabelEncoder().fit(data_set['door_count'])
#data_set['door_count']= door_count_enc.transform(data_set['door_count'])


# In[14]:


#seat_count_enc = LabelEncoder().fit(data_set['seat_count'])
#data_set['seat_count']= seat_count_enc.transform(data_set['seat_count'])


# In[15]:


#manufacture_year_enc = LabelEncoder().fit(data_set['manufacture_year'])
#data_set['manufacture_year']= manufacture_year_enc.transform(data_set['manufacture_year'])


# In[16]:


data_set.head()


# In[17]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
df= pd.DataFrame(data_set)


# In[18]:


data_set=clean_dataset(data_set)


# In[19]:


"""for coloumn in data_set.columns:
    if data_set[coloumn].dtype == type(object):
        le= LabelEncoder()
        data_set[coloumn] = le.fit_transform(data_set[coloumn])"""


# In[20]:


features = data_set.values[:,:9]
features


# In[21]:


outcome= data_set.values[:,9:10]
outcome


# In[22]:


train_features,test_features = train_test_split(features, test_size=0.2)

train_outcome,test_outcome = train_test_split(outcome, test_size=0.2)


# In[23]:


print(len(train_features) , len(test_features))
print(len(train_outcome) , len(test_outcome))


# In[24]:


model=LinearRegression()
model= model.fit(train_features,train_outcome)


# In[25]:


outcome_pred = model.predict(test_features)
outcome_pred


# In[26]:


SSE = (test_outcome - outcome_pred)**2
SSE = sum(SSE)
print("ERROR IS ",SSE)


# In[27]:


def user_interface():
    maker1,model1,mileage1,manufacture_year1,engine_hp1,transmission1,door_count1,seat_count1,fuel_type1 = input("Enter your input with comma seperated values: Maker, Model, Mileage, Manufacture year, Engine_hp, Transmission, Door Count, Seat Count, Fuel type:").split(',')
    new_features = [maker_enc.transform([maker1]),model_enc.transform([model1]),float(mileage1),float(manufacture_year1),float(engine_hp1),transmission_enc.transform([transmission1]),float(door_count1),float(seat_count1),fuel_type_enc.transform([fuel_type1])]
    new_features = np.array(new_features).reshape(1,-1)
    new_outcome = model.predict(new_features)
    new_outcome = new_outcome
    b=np.array(new_outcome)
    return new_features,new_outcome
#.reshape(-1,1)


# In[30]:


new_features,new_outcome = user_interface()
print("Predicted Price: ",new_outcome)
