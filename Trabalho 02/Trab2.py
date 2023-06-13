#!/usr/bin/env python
# coding: utf-8

# # Formatando Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import math


# In[2]:


spreadsheet = pd.read_csv('./orderedDataset.csv')


# In[3]:


spreadsheet.shape


# In[4]:


spreadsheet = spreadsheet.loc[spreadsheet["Valor"] != 0]
spreadsheet.shape


# In[5]:


spreadsheet = spreadsheet.loc[spreadsheet["Valor"] != 1]
spreadsheet.shape


# In[6]:


spreadsheet = spreadsheet.loc[spreadsheet["Valor"] != 6]
spreadsheet.shape


# In[7]:


spreadsheet = spreadsheet.loc[spreadsheet["Valor"] != 100]
spreadsheet.shape


# In[8]:


spreadsheet.head()


# In[9]:


#deixando o dataset em caixa alta, já que o python é case sensitive e os usuários da olx escrevem os nomes dos produtos
#de forma não padronizada
spreadsheet = spreadsheet.apply(lambda x: x.astype(str).str.upper())


# In[10]:


spreadsheet.head()


# In[11]:


spreadsheet = spreadsheet[~spreadsheet['Modelo'].isin(['IPHONE'])]


# In[12]:


spreadsheet = spreadsheet[~spreadsheet['Modelo'].isin(['PRÓ'])]


# In[13]:


spreadsheet = spreadsheet[~spreadsheet['Modelo'].isin(['TROCO'])]


# In[14]:


spreadsheet


# In[15]:


spreadsheet.to_csv('dataset.csv', index=False)


# In[16]:


dataSet_df = pd.read_csv('./dataset.csv')


# # Editando o Dataset de novo

# In[17]:


#iPhone 6
dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & ~dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 6'
dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 6 PLUS'
#iPhone 6S
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & ~dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 6S'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 6S PLUS'
#iPhone 7
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & ~dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 7'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 7 PLUS'
#iPhone 8
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & ~dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 8'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & dataSet_df['Modelo'].str.contains('PLUS'), ['Modelo']] = 'IPHONE 8 PLUS'
#iPhone X
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE X') & ~dataSet_df['Modelo'].str.contains('S'), ['Modelo']] = 'IPHONE X'
#iPhone XR
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE XR'), ['Modelo']] = 'IPHONE XR'
#iPhone XS
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE XS') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE XS'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE XS') & dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE XS MAX'
#iPhone 11
i11 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 11'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 11 PRO'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 11 PRO MAX'
#iPhone 12
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 12'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 12 PRO'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 12 PRO MAX'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 12 MINI'
#iPhone SE
iSE = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE SE'), ['Modelo']] = 'IPHONE SE'
#iPhone 13
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 13'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 13 PRO'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 13 PRO MAX'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 13 MINI'
#iPhone 14
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('PLUS') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 14'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('PLUS') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 14 PRO'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('PLUS') & dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 14 PRO MAX'
dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('PLUS') & ~dataSet_df['Modelo'].str.contains('MAX'), ['Modelo']] = 'IPHONE 14 MINI'
dataSet_df


# ***
# # Medidas de Tendência Central

# In[18]:


#dataset final sem linhas que o codigo não conseguiu remover ou utilizar
dataSet_df = pd.read_csv('./finalDataset.csv')


# ### Média

# In[19]:


print("Média dos valores:", dataSet_df['Valor'].mean())


# In[20]:


print("Média de valor por Modelo:")
#iPhone 6
i6 = dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
i6Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
#iPhone 6S
i6S = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
i6SPlus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
#iPhone 7
i7 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
i7Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
#iPhone 8
i8 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
i8Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].mean()
#iPhone X
i10 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10|IPHONE X') & ~dataSet_df['Modelo'].str.contains('S')]['Valor'].mean()
#iPhone XR
i10R = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10R|IPHONE XR')]['Valor'].mean()
#iPhone XS
i10S = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10S|IPHONE XS') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i10SM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10S|IPHONE XS') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
#iPhone 11
i11 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i11Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i11ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
#iPhone 12
i12 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i12Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i12ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i12Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
#iPhone SE
iSE = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE SE')]['Valor'].mean()
#iPhone 13
i13 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i13Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i13ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i13Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
#iPhone 14
i14 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i14Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i14ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
i14Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].mean()
print("----------Modelos 6----------")
print("iPhone 6:", i6)
print("iPhone 6 Plus:", i6Plus)
print("iPhone 6S:", i6S)
print("iPhone 6S:", i6SPlus)
print("----------Modelos 7----------")
print("iPhone 7:", i7)
print("iPhone 7 Plus:", i7Plus)
#iPhone 8
print("----------Modelos 8----------")
print("iPhone 8:", i8)
print("iPhone 8 Plus:", i8Plus)
#iPhone X
print("----------Modelos X----------")
print("iPhone X:", i10)
#iPhone XR
print("iPhone XR:", i10R)
#iPhone XS
print("iPhone XS:", i10S)
print("iPhone XS Max:", i10SM)
#iPhone 11
print("----------Modelos 11----------")
print("iPhone 11:", i11)
print("iPhone 11 Pro:", i11Pro)
print("iPhone 11 Pro:", i11ProM)
#iPhone 12
print("----------Modelos 12----------")
print("iPhone 12 Mini:", i12Mini)
print("iPhone 12:", i12)
print("iPhone 12 Pro:", i12Pro)
print("iPhone 12 Pro Max:", i12ProM)
#iPhone SE
print("----------Modelos SE----------")
print("iPhone SE:", iSE)
#iPhone 13
print("----------Modelos 13----------")
print("iPhone 13 Mini:", i13Mini)
print("iPhone 13:", i13)
print("iPhone 13 Pro:", i13Pro)
print("iPhone 13 Pro Max:", i13ProM)
#iPhone 14
print("----------Modelos 13----------")
print("iPhone 14 Mini:", i14Mini)
print("iPhone 14:", i14)
print("iPhone 14 Pro:", i14Pro)
print("iPhone 14 Pro Max:", i14ProM)


# ### Mediana

# In[21]:


print("Medianas de valor por Modelo:")
#iPhone 6
i6 = dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
i6Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains(r'\bIPHONE 6\b') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
#iPhone 6S
i6S = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
i6SPlus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 6S') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
#iPhone 7
i7 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
i7Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 7') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
#iPhone 8
i8 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & ~dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
i8Plus = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 8') & dataSet_df['Modelo'].str.contains('PLUS')]['Valor'].median()
#iPhone X
i10 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10|IPHONE X') & ~dataSet_df['Modelo'].str.contains('S')]['Valor'].median()
#iPhone XR
i10R = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10R|IPHONE XR')]['Valor'].median()
#iPhone XS
i10S = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10S|IPHONE XS') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i10SM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 10S|IPHONE XS') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
#iPhone 11
i11 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i11Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i11ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 11') & dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
#iPhone 12
i12 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i12Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i12ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i12Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 12') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
#iPhone SE
iSE = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE SE')]['Valor'].mean()
#iPhone 13
i13 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i13Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i13ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i13Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 13') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
#iPhone 14
i14 = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i14Pro = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i14ProM = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & dataSet_df['Modelo'].str.contains('PRO') & ~dataSet_df['Modelo'].str.contains('MINI') & dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
i14Mini = dataSet_df.loc[dataSet_df['Modelo'].str.contains('IPHONE 14') & ~dataSet_df['Modelo'].str.contains('PRO') & dataSet_df['Modelo'].str.contains('MINI') & ~dataSet_df['Modelo'].str.contains('MAX')]['Valor'].median()
print("----------Modelos 6----------")
print("iPhone 6:", i6)
print("iPhone 6 Plus:", i6Plus)
print("iPhone 6S:", i6S)
print("iPhone 6S:", i6SPlus)
print("----------Modelos 7----------")
print("iPhone 7:", i7)
print("iPhone 7 Plus:", i7Plus)
#iPhone 8
print("----------Modelos 8----------")
print("iPhone 8:", i8)
print("iPhone 8 Plus:", i8Plus)
#iPhone X
print("----------Modelos X----------")
print("iPhone X:", i10)
#iPhone XR
print("iPhone XR:", i10R)
#iPhone XS
print("iPhone XS:", i10S)
print("iPhone XS Max:", i10SM)
#iPhone 11
print("----------Modelos 11----------")
print("iPhone 11:", i11)
print("iPhone 11 Pro:", i11Pro)
print("iPhone 11 Pro:", i11ProM)
#iPhone 12
print("----------Modelos 12----------")
print("iPhone 12 Mini:", i12Mini)
print("iPhone 12:", i12)
print("iPhone 12 Pro:", i12Pro)
print("iPhone 12 Pro Max:", i12ProM)
#iPhone SE
print("----------Modelos SE----------")
print("iPhone SE:", iSE)
#iPhone 13
print("----------Modelos 13----------")
print("iPhone 13 Mini:", i13Mini)
print("iPhone 13:", i13)
print("iPhone 13 Pro:", i13Pro)
print("iPhone 13 Pro Max:", i13ProM)
#iPhone 14
print("----------Modelos 13----------")
print("iPhone 14 Mini:", i14Mini)
print("iPhone 14:", i14)
print("iPhone 14 Pro:", i14Pro)
print("iPhone 14 Pro Max:", i14ProM)


# ***
# # Medidas de Variabilidade

# In[22]:


mediaValor = dataSet_df['Valor'].mean()
mediaValor


# ### Desvio

# In[23]:


desvio = dataSet_df['Valor'].apply(lambda x: x - mediaValor)
desvio


# ### Média do Desvio Absoluto

# In[24]:


acc = 0
for i in range(len(dataSet_df)):
    acc += abs((dataSet_df['Valor'].loc[i] - mediaValor))
acc = acc/len(dataSet_df)
acc 


# ### Variância

# In[25]:


dataSet_df['Valor'].var()


# ### Desvio Padrão

# In[26]:


dataSet_df['Valor'].std()


# ### Boxplot

# In[27]:


dataSet_df.boxplot(column=['Valor'], fontsize='large', figsize=(8,8))


# ***
# # Tabela de Frequência e Histograma

# ### Tabela de Frequência

# In[28]:


frequencia_modelo = dataSet_df['Modelo'].value_counts()
print("Modelo")
print(frequencia_modelo.head(30))

print("\n\n\n")
frequencia_local = dataSet_df['Local'].value_counts()
print("Local")
print(frequencia_local.head(30))


# ### Calculando o número ideal de bins k = ⌈1 + 3,3 log10(tamanho_do_dataset)⌉

# In[29]:


mod_df = pd.read_csv('./modelos.csv')


# In[30]:


k = math.ceil(1 + 3.3 * math.log10(mod_df.size))
k


# In[31]:


import matplotlib.pyplot as plt
plt.hist(mod_df['Valor'], bins=k)
plt.show()
plt.hist(mod_df['Modelo'], bins=k)
plt.show()


# ### Calculando o número ideal de bins k = sqrt(tamanho_do_dataset)

# In[32]:


k = int(math.sqrt(mod_df.size))
k


# In[33]:


import matplotlib.pyplot as plt
plt.hist(mod_df['Valor'], bins=k)
plt.show()
plt.hist(mod_df['Modelo'], bins=k)
plt.show()


# In[34]:


dataSet_df.to_csv('dataset.csv', index=False)


# # Correlações

# In[35]:


dataSet_df = pd.read_csv('./finalDataset.csv')


# In[36]:


mod_df = pd.read_csv('./modelos.csv')


# In[37]:


mod_df.apply(lambda x: x.factorize()[0]).corr()


# In[38]:


import seaborn as sns
import pylab as py

sns.heatmap(pd.crosstab(mod_df.Modelo,mod_df.Local))
py.show()
sns.heatmap(pd.crosstab(mod_df.Valor,mod_df.Local))
py.show()


# In[39]:


sns.heatmap(pd.crosstab(mod_df.Valor,mod_df.Modelo))
py.show()


# # QQ-Plot

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
import statsmodels.api as sm
import pylab as py
import random


# In[41]:


housefly_z_df = stats.zscore(dataSet_df['Valor'])
housefly_z_df


# In[42]:


sm.qqplot(housefly_z_df, line ='45') 
py.show() 


# # Teste de Normalidade

# In[43]:


# Computando a Frequência Absoluta (Fabs) e colocando o resultado em um novo dataframe

table_df = dataSet_df.groupby(['Valor']).size().reset_index(name='Fabs')
table_df


# In[44]:


# Renomeando a coluna de 0 para X_i (lê-se: x índice i)

newcols = {
    'Valor': 'Xi'
}
table_df.rename(columns=newcols, inplace=True)
table_df


# In[45]:


# Calculando a Frequência Acumulada (Fac)

table_df['Fac'] = table_df['Fabs'].cumsum()
table_df


# In[46]:


# Calculando a coluna Fracionária: Total acumulado dividido pelo valor máximo total

table_df['Frac'] = table_df['Fac']/table_df['Fac'].max()
table_df


# In[47]:


# Calculando a média 

mean = dataSet_df['Valor'].mean()
mean


# In[48]:


# Calculando o desvio padrão

std = dataSet_df['Valor'].std()
std


# In[49]:


# Agora vamos normalizar os dados de X_i para Z_i
# A normalização Z é dada por subtrair de cada elemento a média e dividir pelo desvio padrão

table_df['Zi'] = table_df['Xi'].apply(lambda x: (x - mean)/std)
table_df


# In[50]:


import scipy.special as scsp
def zScoreToPvalue(z):
    # Retornar p-value a partir do z-score
    return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))


# In[51]:


table_df['FracEsp'] = table_df['Zi'].apply(lambda x: zScoreToPvalue(x))
table_df


# In[52]:


# Result1 = FracEsp - Frac
table_df['D_negativo'] = abs(table_df['FracEsp']-table_df['Frac'])
table_df


# In[53]:


# Criando uma coluna de zeros 
table_df['D_positivo'] = 0
table_df


# In[54]:


for i in range(table_df['Frac'].shape[0]):
    if i > 0:
        table_df['D_positivo'].iloc[i] = table_df['FracEsp'].iloc[i] - table_df['Frac'].iloc[i-1]
    else:
        table_df['D_positivo'].iloc[i] = table_df['FracEsp'].iloc[i]


# In[55]:


table_df


# In[56]:


# Calcular o máximo valor da coluna Result1 e depois o máximo da coluna Result 2
# E, por fim, retornar o maior dos dois
D = ( table_df[['D_negativo','D_positivo']].max() ).max()
D


# In[57]:


from scipy.stats import ksone

def ks_critical_value(n_trials, alpha):
    return ksone.ppf(1-alpha/2, n_trials)


# In[58]:


# n-trials: quantidade de dados
# alpha: Um bom valor para o nível de significância do teste é com um alfa = 0,05 para assegurar 95% de confiança
p_value = ks_critical_value(dataSet_df.shape[0], 0.05)
p_value


# In[59]:


if D < p_value:
    print('Os dados seguem uma distribuição normal')
else:
    print('Os dados não seguem uma distribuição normal')


# In[60]:


y_std = stats.zscore(dataSet_df['Valor'])
print(y_std)
D, p = stats.kstest(y_std, 'norm', alternative='greater')
print("D = {} | p = {}".format(D, p))
if D < p:
    print('Os dados seguem uma distribuição normal.')
else:
    print('Os dados não seguem uma distribuição normal.')


# # Best Fit Distribution

# In[61]:


import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              't',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max']


# In[63]:


y_std_valor = stats.zscore(dataSet_df['Valor'])


# In[64]:


def check_distribution(dist_names, y_std_valor):
    
    p_values = []
    distance = []
    D_less_p = []
    
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std_valor)

        if distribution != "norm":
            D, p = scipy.stats.kstest(y_std_valor, distribution, args=param)
        else:
            D, p = scipy.stats.kstest(y_std_valor, distribution,  alternative='greater')
            
        #p = np.around(p, 5)
        p_values.append(p)    
        
        #D = np.around(D, 5)
        distance.append(D)    
        
        if D<p: 
            D_less_p.append("yes") 
        else: 
            D_less_p.append("no")

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['Distance'] = distance
    results['p_value'] = p_values
    results['D<p'] = D_less_p
    
    results.sort_values(['p_value'], ascending=False, inplace=True)


    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)


# In[65]:


check_distribution(dist_names, y_std_valor)


# In[66]:


plt.hist(y_std_valor,bins=6)
plt.show()


# In[ ]:




