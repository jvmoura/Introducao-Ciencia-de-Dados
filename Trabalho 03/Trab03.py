#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import math


# ## Medidas de Tendência Central

# In[2]:


df = pd.read_csv('./dataset.csv')
Gdf = pd.read_csv('./dataGraph.csv')


# ### Média

# In[3]:


print("Média dos preços:", df['Valor'].mean())


# In[4]:


print("Média dos anos de lançamento:", df['Ano'].mean())


# ### Mediana

# In[5]:


print("Mediana:", df['Valor'].median())


# ## Medidas de Dispersão

# ### Desvio

# In[6]:


media = df['Valor'].mean()
# Calculando o Desvio
d = df['Valor'].apply(lambda x: x - media)
d


# ### Média do Desvio Absoluto

# In[7]:


acc = 0
for i in range(len(df)):
    acc += abs((df['Valor'].loc[i] - media))
acc = acc/len(df)
acc 


# ### Variância

# In[8]:


df['Valor'].var()


# ### Desvio Padrão

# In[9]:


df['Valor'].std()


# ## Boxplot

# In[10]:


#Boxplot feito para cada modelo de aparelho
Gdf.boxplot(by='Modelo', column=['Valor'], fontsize='large', figsize=(15,15))


# In[11]:


#Boxplot geral
Gdf.boxplot(column=['Valor'], fontsize='large', figsize=(8,8))


# In[12]:


# Primeiro Quartil
Q1 = np.percentile(df['Valor'], 25,
                   interpolation = 'midpoint')

#Terceiro Quartil
Q3 = np.percentile(df['Valor'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1
IQR


# In[13]:


#Limite superior
upper = Q3+1.5*IQR
upper


# In[14]:


#Limite inferior
lower = Q1-1.5*IQR
lower


# In[15]:


#Acima do limite superior
outupper = df['Valor'] >= upper
outupper
print("Upper bound:\n",outupper)
print(np.where(outupper))
 
#Abaixo do limite superior
outlower = df['Valor'] <= lower
outlower
print("Lower bound:\n", outlower)
print(np.where(outlower))


# ## Z-Score

# In[16]:


from scipy import stats
z = stats.zscore(df['Valor'])
print(z)


# In[17]:


threshold = 3
 
# Position of the outlier
outlier = np.where(np.abs(z) > 3)
outlier


# In[18]:


df.drop('Modelo', axis=1, inplace=True)
df


# In[19]:


new_df = df[(df['Valor'] < upper) & (df['Valor'] > lower)]
new_df


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['Valor'])
plt.subplot(2,2,2)
sns.boxplot(df['Valor'])
plt.subplot(2,2,3)
sns.distplot(new_df['Valor'])
plt.subplot(2,2,4)
sns.boxplot(new_df['Valor'])
plt.show()


# # Isolation Forest

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ### Testando a Regressão Linear Sem a Remoção de Outliers

# In[23]:


df = pd.read_csv('./dataGraph.csv')
dfGraph = pd.read_csv('./dataGraph.csv')


# In[24]:


#Função que transforma elementos não-numéricos em numéricos
dfGraph.fillna(0, inplace=True)

columns = df.columns.values
def handle_non_numerical_data(df):
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

dfGraph = handle_non_numerical_data(dfGraph)
print('----------Antes----------')
print(df.head())
print('\n----------Depois----------')
print(dfGraph.head())


# In[25]:


#Recuperar Array
data = dfGraph.values
# Separar em elementos de input e output
X, y = data[:, 1:5], data[:, 0]
# Resume o shape do dataset
print('Shape X; Shape Y')
print(X.shape, y.shape)
print('\n')
# Separa em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# Resume o shape do treino e teste
print('Treino X; Teste X')
print(X_train.shape, X_test.shape)
print('\n')
print('Treino Y; Teste Y')
print(y_train.shape, y_test.shape)


# In[26]:


dfGraph


# In[27]:


y


# In[28]:


X


# In[29]:


# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# Avalia o modelo
yhat = model.predict(X_test)
# Avalia previsões
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# ### Detectando Outliers com o Método Isolation Forest

# In[30]:


dfGraph = handle_non_numerical_data(dfGraph)
# Identifica outliers no dataset treino
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)


# In[31]:


model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(X_train,y_train)


# In[32]:


dfGraph = dfGraph.iloc[:,0:4]
dfGraph


# In[33]:


dfGraph['scores']=model.decision_function(dfGraph.iloc[:,1:4])


# In[34]:


dfGraph


# In[35]:


X = dfGraph.iloc[:,1:4]
X


# In[36]:


y = dfGraph.iloc[:,0]
y


# In[37]:


dfGraph['anomaly']=model.predict(X)


# In[38]:


anomaly=dfGraph.loc[dfGraph['anomaly']==-1]
anomaly_index=list(anomaly.index)


# In[39]:


anomaly


# ### Removendo os Outliers

# In[40]:


#X = X.reshape(-1, X.shape[-1])
# select all rows that are not outliers

mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]


# In[41]:


X_train.shape


# In[42]:


y_train.shape


# In[43]:


dfGraph = pd.read_csv('./dataGraph.csv')
dfGraph = handle_non_numerical_data(dfGraph)
dfGraph


# In[44]:


# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
# retrieve the array
data = dfGraph.values
# split into input and output elements
X, y = data[:, 1:5], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# ### Minimum Covariance Determinant (MCD)

# In[45]:


df = pd.read_csv('./dataGraph.csv')
# show data in a scatterplot
plt.figure(figsize=(13,6))
plt.scatter(df["Modelo"], df["Valor"], color = "r")
plt.grid()


# In[46]:


# convert dataframe to arrays
dfGraph = handle_non_numerical_data(dfGraph)
data = dfGraph[['Modelo', 'Valor']].values


# In[47]:


from sklearn.covariance import EllipticEnvelope
# instantiate model
model1 = EllipticEnvelope(contamination = 0.1) 
# fit model
model1.fit(data)


# In[48]:


# new data for prediction (data needs to be in arrays)
new_data = np.array([[10,10], [1,1], [1,1], [1,1]])
# predict on new data 
pred1 = model1.predict(new_data)
print(pred1)


# ### Testando a Regressão Linear Após a Remoção de Outliers Usando Minimum Covariance Determinant (MCD)

# In[49]:


# retrieve the array
data = dfGraph.values
# split into input and output elements
X, y = data[:, 1:5], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)


# In[50]:


# identify outliers in the training dataset
from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)


# In[51]:


y_train


# In[52]:


# evaluate model performance with outliers removed using elliptical envelope
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
# retrieve the array
data = dfGraph.values
# split into input and output elements
X, y = data[:, 1:5], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# ### Local Outlier Factor (LOF)

# In[53]:


# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor


# In[54]:


df = pd.read_csv('./dataGraph.csv')
df


# In[55]:


# plot data points
plt.figure(figsize=(13,6))
plt.scatter(df["Modelo"], df["Valor"], color = "b", s = 30)
plt.grid()


# In[56]:


dfGraph = handle_non_numerical_data(dfGraph)
# model specification
model1 = LocalOutlierFactor(n_neighbors = 2, metric = "manhattan", contamination = 0.02)
# model fitting
y_pred = model1.fit_predict(dfGraph)


# In[57]:


y_pred


# In[58]:


# filter outlier index
#outlier_index = where(y_pred == -1) # negative values are outliers and positives inliers
# filter outlier values
outlier_index = 4


# In[59]:


df = pd.read_csv('./dataGraph.csv')
outlier_values = df.iloc[outlier_index]
# plot data
plt.figure(figsize=(13,6))
plt.scatter(df["Modelo"], df["Valor"], color = "b", s = 65)
# plot outlier values
plt.scatter(outlier_values["Modelo"], outlier_values["Valor"], color = "r")


# ### Testando a Regressão Linear Após a Remoção de Outliers Usando Local Outlier Factor (LOF)

# In[60]:


#convertando valores nao-numericos para numericos
dfGraph = handle_non_numerical_data(dfGraph)
# retrieve the array
data = dfGraph.values
# split into input and output elements
X, y = data[:, 1:5], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)


# In[61]:


dfGraph


# In[62]:


# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)


# In[63]:


# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# ## One-Class SVM

# ### Testando a Regressão Linear Após a Remoção de Outliers Usando Local Outlier Factor (LOF)

# In[64]:


from sklearn.svm import OneClassSVM


# In[65]:


# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)


# In[66]:


# evaluate model performance with outliers removed using one class SVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# retrieve the array
data = dfGraph.values
# split into input and output elements
X, y = data[:, 1:5], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# In[ ]:




