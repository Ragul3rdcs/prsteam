#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
from collections import Counter as c
import matplotlib.pyplot as plt
import seaborn as sns 
import missingno as msno
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LableEncoder
from sklearn.linear_model import logisticRegression
import pickle


# In[67]:


data=pd.read_csv("D:\\NMGTR\kidney_disease.csv")
data.head()


# In[ ]:


data.columns


# In[ ]:


data.columns=['age','blood_pressure','specfic_gravity','albumin',
              'suger','red_blood_cells','pus_cell','pus_cell_clumps','bacteria',
              'blood glucous random','blood_urea','serum_ creatinine','sodium','potassium',
              'hemoglobin','packed_cell_valume','white_blood_cell_count','red_blood_cell_count',
              'hypertension','diabetesmellitus','coronary_artery_disease','appetite',
              'pedal_edema','anemia','class']
data.columns


# In[68]:


data.info()


# In[69]:


data.isnull().any()


# In[73]:


data['blood glucose random'].fillna(data['blood glucose random'].mean(),inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace=True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace=True)
data['sodium'].fillna(data['sodium'].mean(),inplace=True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace=True)


# In[74]:


data['age'].fillna(data['age'].mode()[0],inplace=True)
data['hypertension'].fillna(data['hypertention'].mode()[0],inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0],inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0],inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0],inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0],inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0],inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0],inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0],inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0],inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0],inplace=True)
data['diabatesmellitus'].fillna(data['diabatesmellitus'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0],inplace=True)


# In[75]:


catcols=set(data.dtypes[data.dtypes=='0'].index.values)# only fetch the object type column
print(catcols)


# In[76]:


for i in catcols:
    print("Columns:",i)
    print(c(data[i])) # using counter for number of classes in the column
    print('*'*120+'\n')


# In[77]:


catcols.remove('red_blood_cells_count') # remove is used for removing a particular column
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')
print(catcols)


# In[ ]:


#'specific_gravity','sugar'(as these columns are numerical it is removed)
catcols=['anemia','pedal_edema','appetite','class','coronary_artery_disease','diabetesmellit'
        'hypertension','pus_cell','pus_cell_clumps','red_blood_cells']#only considered the text class columns


# In[78]:


from sklearn.preprocessing import LabelEncoder # importing the LableEncoding fropm sklearn
for i in catcols: #looping through all the categorical columns
    print("LABEL ENCODING OF:",i)
    LEi=LabelEncoder() #creting on object of LabelEncoder
    print(c(data[i]) #getting the classes values before transformation
    data[i]=LEi.fit_transform(data[i]) #transforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print("*"*100)


# In[79]:


contcols=set(data.dtypes[data.dtypes!='0'].index.values)# only fetch the float and int type columns
#contcols=pd.DataFrame(data,column=contcols)
print(contcols)


# In[80]:


for i in contcols:
    print("Continous Columns:",i)
    print(c(data[i]))
    print('*'*120+'\n')


# In[81]:


contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)


# In[82]:


contcols.add('red_blood_cell_count')# using add we can add the column
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)


# In[83]:


catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)


# In[84]:


data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')# replacing \tno wi
c(data['coronary_artery_disease'])


# In[85]:


data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\type':'yes','yes':''})
c(data['diabetesmellitus'])


# In[86]:


data.describe()


# In[87]:


sns.distplot(data.age)


# In[92]:


import matplotlib.pyplot as plt
figure=plt.figure(figsize=(5,5))
plt.scatter(data['age'],data['blood_pressure'],color='blue')
plt.xlabel('age')
plt.ylabel('blood_pressure')
polt.title("age VS blood Scatter Plot")


# In[93]:


plt.figure(figsize=(20,15),facecolor='white')
plotnumber=1

for column in contcols:
    if plotnumber<=11:
        ax=plt.subplot(3,4,plotnumber)
        plt.scatter(data['age'],data[column])
        plt.xlabel(column,fontsize=20)
        
        plotnumber+=1
plt.show()


# In[94]:


f,sx=plt.sublots(figsize=(18,10))
sns.heartmap(data,corr(),annot=true,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# In[95]:


sns.countplot(data['class'])


# In[96]:


from sklearn.preprocessing import Standerdscaler
sc=StanederdScaler(
x_bal=sc.fit_transformx())


# In[97]:


selcols=['red_blood_cells','pus_cells','blood glucose random','blood_urea',
'pedal_edema','anemia','diabetesmellitus','coranaory_artery_disease']
x=pd.DataFrame(data,coloumns=selcous)
y=pd.DataFrame(data,columns=['class'])
print(x.shape)
print(y.shape


# In[98]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import dense


# In[99]:


classification=Sequential()
classification.add(Denes(30.activation='relu'))
classification.add(Denes(128.activation='relu'))
classification.add(Denes(64.activation='relu'))
classification.add(Denes(32.activation='relu'))
classification.add(Denes(1.activation='relu'))


# In[100]:


calsification.combile(optimizer='admin',loss='binary_crossentqry',metrics=['accuracy'])



classification.fit(x_train,y_train.batch_size=10,validation_split=0.2,epochs=100)


# In[101]:


from sklearn.ensemble import RandamForestClassifier
rfc=RandamForest(n_estimators=10,criterion='entroy')


# In[ ]:


rfc.fit(x_train)


y_predict=rfc.predict(x_test)


# In[102]:


from sklearn.linear_model import LogicticsRegestration
lgr=LogisticRegestration()
lgr.fit(x_train,y_train)



LogiesticRegistration()


# In[103]:


from sklearn.metrics import accuracy_score,classification_report
y_predict=lgr.predict(x_test)


# In[104]:


classification.save("{ckd.hs}")
y_pred=classification.predict(x_test)


# In[105]:


y_pred=(y_pred>0.5)
y_pred


# In[106]:


def predict_exit(sample_value):
    
    sample_value=np.array(sample_value)
    
    sample_value=sample_value.reshape(1.-1)
    
    sample_value=sc.tranceform(sample_values)


# In[107]:


test=classification.preduct([[1,1,121,000000,36.0,0,0,1,0,]])
if test==1;
    print('prediction:high change of CKD!')
else;
print('prediction:low change of SKD.')


# In[109]:


from sklearn import modle_selection 


# In[ ]:


from sklearn.matrics import confusion_matrix
cm


# In[110]:


plt.figure(figsize=(8,6))
sns.heatmap(cm.cmp='blues',annot=true,xticklabels=['no ckd,'ckd],yticklabls=['no ckd','ckd'])
plt.xlabl;e('predicted values')
        


# In[ ]:




