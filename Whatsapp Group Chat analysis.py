#!/usr/bin/env python
# coding: utf-8

# # Whatsapp Group Chat Analysis 
# 

# ### Importing required libraries

# In[1]:


import numpy as np
import pandas as pd
import emoji
import matplotlib.pyplot as plt


# In[44]:


import regex
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# ### Reading the group chat file 

# In[3]:


a=pd.read_csv(r"C:\Users\hp\Downloads\WhatsApp Chat with Brahma 2019-23.txt", delimiter = "\t", header = None, names = ['text'])


# In[4]:


b=a.copy()


# ### Data Cleaning

# In[5]:


a[['datetime_str', 'text_2']] = a["text"].str.split(" - ", 1, expand=True)
a["datetime"] = pd.to_datetime(a["datetime_str"],
                               format="%d/%m/%Y, %I:%M %p",
                               errors='coerce')
a[['sender', 'text_message']] = a['text_2'].str.split(': ', 1, expand=True)
a['first_name'] = a['sender'].str.split(' ', expand=True)[0]
a['last_name'] = a['sender'].str.split(' ', 1, expand=True)[1]
a['sender'] = a['first_name'].where(
    a['last_name'].isnull(), a['first_name'] + ' ' + a['last_name'].str[:])


# In[6]:


a.tail()


# In[7]:


a['date_col'] = pd.to_datetime(a['datetime_str'])


# In[8]:


a['Date'] = a['date_col'].dt.date
a['Time'] = a['date_col'].dt.strftime('%H:%M')
a.head()


# In[9]:


a.tail()


# In[10]:


a.drop(
    ['text', 'datetime_str', 'text_2', 'datetime', 'first_name', 'last_name'],
    axis=1,
    inplace=True)


# In[11]:


a


# ## Data Visualization

# ### Heatmap for Hour of Day v/s Day of week

# In[12]:


a['day_of_week'] = a['date_col'].dt.dayofweek + 1
a['hour_of_day'] = a['date_col'].dt.hour

# Create new Dataframe containing data counts
heatmap_data = a.groupby(['day_of_week', 'hour_of_day']).size()
heatmap_data = heatmap_data.unstack()

# Create heatmap
plt.pcolor(heatmap_data, cmap='Reds')
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.colorbar()
plt.show()


# ### Bar Chart for sender message count

# In[46]:


sender_count_series = a.groupby(['sender']).size().sort_values(ascending=False).head(20)

# Create sender counts series as a DataFrame
sender_count_df = pd.DataFrame(sender_count_series)

# Reset index in order to name columns correctly
sender_count_df = sender_count_df.reset_index()
sender_count_df.columns = ['sender', 'count']

# Plot bar chart with sender message counts
plt.figure(figsize=(15, 5))
New_Colors = ['red','green','blue','purple','orange','maroon','black','cyan','violet','navy']
plt.bar(sender_count_df['sender'], sender_count_df['count'], color = New_Colors )
plt.xlabel("Sender")
plt.ylabel("Message Count")
plt.xticks(rotation=30, ha="right")
plt.show()


# In[15]:


author_value_counts = a['sender'].value_counts() 
top_10_author_value_counts = author_value_counts.head(10) 
top_10_author_value_counts.plot.barh(color = New_Colors)


# In[47]:


top_10_author_value_counts


# #### Manipulation : Replacing null message and spamming blank space message with 'Jai Brahma'

# In[17]:


a['text_message'] = a['text_message'].replace(np.nan, 'Jai Brahma')


# ### Most used Emoji's 

# In[31]:


emojis = pd.DataFrame(columns=['sender','emoji','date_col'])

# Loop through all messages in the DataFrame
for sender, message, date_col in zip(a.sender, a.text_message, a.date_col):

  # Split out each word in each message
  message_split = list(message)
 
  # Loop through each word in split message
  for character in message_split:
 
    # If the word is an emoji
    if character in emoji.UNICODE_EMOJI and character != "\U0001f3fc":
 
      # Add each emoji to the DataFrame
      emojis = emojis.append({'sender' : sender, 'emoji' : character, 'date_col' : date_col}, ignore_index=True)

# Display top n most popular emojis
emojis.groupby(['emoji']).size().sort_values(ascending=False).head(10)


# ### Group Wise Stats

# In[20]:


import re


# In[49]:


def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list


total_messages = a.shape[0]
media_messages = a[a['text_message'] == '<Media omitted>'].shape[0]
a["emoji"] = a["text_message"].apply(split_count)
emojis = sum(a['emoji'].str.len())
URLPATTERN = r'(https?://\S+)'
a['urlcount'] = a.text_message.apply(
    lambda x: re.findall(URLPATTERN, x)).str.len()
links = np.sum(a.urlcount)


# #### Total messages in group

# In[50]:


total_messages


# #### Total media messages

# In[51]:


media_messages


# #### Total emojis used

# In[52]:


emojis


# #### Total links sent

# In[53]:


links


# ### Omitting Media messages

# In[26]:


media_messages_df = a[a['text_message'] == '<Media omitted>']
messages_df = a.drop(media_messages_df.index)


# In[27]:


messages_df['Letter_Count'] = messages_df['text_message'].apply(
    lambda s: len(s))
messages_df['Word_Count'] = messages_df['text_message'].apply(
    lambda s: len(s.split(' ')))


# In[28]:


messages_df


# ### Sender wise Stats

# In[29]:


l = messages_df.sender.unique()

for i in range(len(l)):
    # Filtering out messages of particular user
    req_df = messages_df[messages_df["sender"] == l[i]]
    # req_df will contain messages of only one particular user
    print(f'Stats of {l[i]} -')
    # shape will print number of rows which indirectly means the number of messages
    print('Messages Sent', req_df.shape[0])
    #Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
    words_per_message = (np.sum(req_df['Word_Count'])) / req_df.shape[0]
    print('Words per message', words_per_message)
    #media conists of media messages
    media = media_messages_df[media_messages_df['sender'] == l[i]].shape[0]
    print('Media Messages Sent', media)
    # emojis conists of total emojis
    emojis = sum(req_df['emoji'].str.len())
    print('Emojis Sent', emojis)
    #links consist of total links
    links = sum(req_df["urlcount"])
    print('Links Sent', links)
    print()


# In[55]:


#emojis.groupby(['emoji']).size().sort_values(ascending=False)


# ### Total count of words in all messages

# In[33]:


text = " ".join(review for review in messages_df.text_message)
print ("There are {} words in all the messages.".format(len(text)))


# ### Wordcloud of most used words

# In[34]:


from wordcloud import WordCloud


# In[35]:


wordcloud = WordCloud(background_color="white").generate(text)
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Most happening day

# In[36]:


date_df = messages_df.groupby("Date").sum()
date_df.reset_index(inplace=True)
date_df.head()


# In[37]:


messages_df['Date'].value_counts().head(10).plot.barh(color = New_Colors)
plt.xlabel('Number of Messages')
plt.ylabel('Date')


# ### Most happening time

# In[38]:


messages_df['Time'].value_counts().head(10).plot.barh(color = New_Colors)
plt.xlabel('Number of messages')
plt.ylabel('Time')


# ### Number of words used by top 10 Senders

# In[103]:


total_word_count_grouped_by_author = messages_df[['sender', 'Word_Count'
                                                  ]].groupby('sender').sum()
sorted_total_word_count_grouped_by_author = total_word_count_grouped_by_author.sort_values(
    'Word_Count', ascending=False)
top_10_sorted_total_word_count_grouped_by_author = sorted_total_word_count_grouped_by_author.head(
    10)
top_10_sorted_total_word_count_grouped_by_author.plot.barh(color='navy')
plt.xlabel('Number of Words')
plt.ylabel('sender')


# In[54]:


#req_df


# ### Top words used by all group members

# In[45]:


l = messages_df.sender.unique()
for i in range(len(l)):
    dummy_df = messages_df[messages_df['sender'] == l[i]]
    text = " ".join(review for review in dummy_df.text_message)
    stopwords = set(STOPWORDS)
    stopwords.update(["kya", "message", "deleted", "hai", "tha", "ki"])
    
    print('Sender', l[i])
    
    wordcloud = WordCloud(background_color="white").generate(text)
   
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[ ]:





# In[ ]:




