#!/usr/bin/env python
# coding: utf-8

# ## Book Recommendation

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


books = pd.read_csv("C:\\Users\\jashj\\OneDrive\\Desktop\\CLUSTERING\\Books.csv")
ratings = pd.read_csv("C:\\Users\\jashj\\OneDrive\\Desktop\\CLUSTERING\\Ratings.csv")
users = pd.read_csv("C:\\Users\\jashj\\OneDrive\\Desktop\\CLUSTERING\\Users.csv")


# In[3]:


books.head()


# In[4]:


ratings.head()


# In[5]:


users.head()


# In[6]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


ratings.isnull().sum()


# ## Popularity based recommender system

# In[10]:


ratings_with_name = ratings.merge(books,on="ISBN")


# In[11]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating' : 'Num_ratings'},inplace= True)
num_rating_df


# In[12]:


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating' : 'Avg_ratings'},inplace= True)
avg_rating_df


# In[13]:


popular_df = num_rating_df.merge(avg_rating_df, on = 'Book-Title')
popular_df


# In[14]:


popular_df = popular_df[popular_df['Num_ratings']>=250].sort_values('Avg_ratings', ascending=False).head(50)


# In[15]:


popular_df


# In[16]:


popular_df.merge(books,on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','Num_ratings','Avg_ratings']]


# ## Collaborative Filtering Based Recommender System

# #### user > 200    considering user only who have rated more than 200 books
# #### books > 50    considering books which have minimum 50 rating on it

# In[17]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index


# In[19]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[28]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>= 50
famous_books = y[y].index

# y[y] boolan indexing


# In[32]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[33]:


final_ratings.drop_duplicates()


# In[37]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[39]:


pt.fillna(0,inplace=True)


# In[40]:


pt


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


similarity_score = cosine_similarity(pt)


# In[45]:


similarity_score


# In[54]:


def recommend(book_name):
    #index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    for i in similar_items:
        print(pt.index[i[0]])


# In[57]:


recommend('Message in a Bottle')


# In[ ]:




