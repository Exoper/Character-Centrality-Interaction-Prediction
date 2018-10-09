
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
import community
import networkx as nx
import warnings
import numpy as np
warnings.filterwarnings('ignore')
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from numpy import array
#-from numpy import argmax
from catboost import CatBoostClassifier, FeaturesData, Pool
from catboost import CatBoostRegressor
import seaborn as sns


# In[88]:


book1 = pd.read_csv('/home/exoper/Documents/data/asoiaf-book1-edges.csv')
book2 = pd.read_csv('/home/exoper/Documents/data/asoiaf-book2-edges.csv')
book3 = pd.read_csv('/home/exoper/Documents/data/asoiaf-book3-edges.csv')
book4 = pd.read_csv('/home/exoper/Documents/data/asoiaf-book4-edges.csv')
book5 = pd.read_csv('/home/exoper/Documents/data/asoiaf-book5-edges.csv')
print(book1.shape)
book1.head(10)


# In[3]:



G_book1 = nx.Graph()
G_book2 = nx.Graph()
G_book3 = nx.Graph()
G_book4 = nx.Graph()
G_book5 = nx.Graph()
books = [G_book1, G_book2, G_book3, G_book4, G_book5]


# In[4]:


for row in book1.iterrows():
    G_book1.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book2.iterrows():
    G_book2.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book3.iterrows():
    G_book3.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book4.iterrows():
    G_book4.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book5.iterrows():
    G_book5.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])


# ## Visualization of the Network
# 

# In[5]:


graph_pos = nx.spring_layout(G_book1)
nx.draw_networkx_nodes(G_book1, graph_pos, node_size=12, node_color='red', alpha=0.3)
nx.draw_networkx_edges(G_book1, graph_pos  ,edge_color='blue')
nx.draw_networkx_labels(G_book1, graph_pos, font_size=8, font_family='sans-serif')
plt.savefig("plot.png", dpi=2000)
plt.savefig("plot.pdf")
plt.gcf().set_size_inches(18, 16)
plt.show()
#nx.draw(G_book1 , with_labels=True )
# nx.draw(G_book2 , with_labels=True )
# nx.draw(G_book3 , with_labels=True )
# nx.draw(G_book4 , with_labels=True )
# nx.draw(G_book5 , with_labels=True )


# ## Centrality 

# In[6]:


b1=sorted(nx.betweenness_centrality(G_book1 , weight='weight').items() ,key = lambda x:x[1] , reverse=True)[:20]
print(b1)


# In[89]:


b2=sorted(nx.betweenness_centrality(G_book2 , weight='weight').items() ,key = lambda x:x[1] , reverse=True)[:20]
#print(b2)


# In[8]:


b3=sorted(nx.betweenness_centrality(G_book3 , weight='weight').items() ,key = lambda x:x[1] , reverse=True)[:20]


# In[9]:


b4=sorted(nx.betweenness_centrality(G_book4 , weight='weight').items() ,key = lambda x:x[1] , reverse=True)[:20]


# In[10]:


b5=sorted(nx.betweenness_centrality(G_book5 , weight='weight').items() ,key = lambda x:x[1] , reverse=True)[:20]


# ## Shuffling the dataset

# In[11]:


filepaths = ['/home/exoper/Documents/data/asoiaf-book1-edges.csv', '/home/exoper/Documents/data/asoiaf-book2-edges.csv','/home/exoper/Documents/data/asoiaf-book3-edges.csv','/home/exoper/Documents/data/asoiaf-book4-edges.csv','/home/exoper/Documents/data/asoiaf-book5-edges.csv']
df = pd.concat(map(pd.read_csv, filepaths))
t = df.Target.unique()
s = df.Source.unique()
dt = df.sample(frac=1)


# In[12]:


df.head(10)


# In[13]:


dt.head(10)


# ## Dropping Unecessary Columns

# In[14]:


dt = dt.drop(columns='Type')


# In[15]:


c = df.groupby('Source')['weight'].sum()


# In[16]:


c1 = c.sort_values(ascending=False)[5:]
c2 = c1.index.tolist()
#print(c2)
   


# ## Limiting to Central Characters

# In[32]:


ds = dt[~dt['Source'].isin(c2)]
print(ds.shape)

dw = ds.sample(frac=1)
dw.sort_values(by =['weight'] ,ascending = False)


# ## CatBoost

# In[33]:


X = dw.drop(['weight'], axis=1)
y = dw.weight
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8,random_state=42)


# In[34]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]


# In[35]:


model=CatBoostRegressor(iterations=1,depth=10, learning_rate=0.1, loss_function='RMSE',use_best_model=True)
model.fit(X_train, y_train,cat_features=categorical_features_indices , eval_set=(X_validation,y_validation))


# In[84]:


da = ds.sample(frac=1)
def conv(s):
    if s < 15:
        return 0
    elif s>=15 and s<40:
        return 1
    elif s>=40 and s<80:
        return 2
    else:
        return 3
da.weight = da.weight.map(conv)


# In[85]:


X = da.drop(['weight'], axis=1)
y = da.weight
from sklearn.model_selection import train_test_split
X_t, X_val, y_t, y_val = train_test_split(X, y, train_size=0.8,random_state=42)
categorical_features_indices = np.where(X.dtypes != np.float)[0]
model=CatBoostRegressor(iterations=10, depth=9, learning_rate=0.1, loss_function='RMSE')
model.fit(X_t, y_t,cat_features=categorical_features_indices,eval_set=(X_val, y_val))

