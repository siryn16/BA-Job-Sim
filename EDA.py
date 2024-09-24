#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/bendh/Desktop/data science/JN/customer_booking.csv", encoding="ISO-8859-1")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# ## Sales Channel

# In[6]:


#sales channel booking was made on (internet/phone call)
internet_sales_channel = df.sales_channel.value_counts().values[0] / df.sales_channel.count() * 100
phone_sales_channel = df.sales_channel.value_counts().values[1] / df.sales_channel.count() * 100


# In[7]:


print(f"Percentage of booking made through internet:{internet_sales_channel}%")
print(f"Percentage of booking made through phone calls:{phone_sales_channel}%")


# ## Trip type

# In[8]:


#type of the trip (Round trip/One way/ Circle trip)
round_trip_type = df.trip_type.value_counts().values[0] / df.trip_type.count() * 100
one_way_trip_type = df.trip_type.value_counts().values[1] / df.trip_type.count() * 100
circle_trip_type = df.trip_type.value_counts().values[2] / df.trip_type.count() * 100


# In[9]:


print(f"Percentage of booking a round trip type:{round_trip_type}%")
print(f"Percentage of booking a one way trip type:{one_way_trip_type}%")
print(f"Percentage of booking a circle trip type:{circle_trip_type}%")


# ## Purchase Lead

# In[10]:


#number of days between travel date and booking date

# Set the figure size
plt.figure(figsize=(15,5))

# Create the histogram with KDE
sns.histplot(data=df, x="purchase_lead", binwidth=20, kde=True)

# Set the x-axis label with units (days)
plt.xlabel("Purchase Lead Time", fontsize=10)

# Set the y-axis label with a description (number of people)
plt.ylabel("Bookings Count", fontsize=10)

# Show the plot
plt.show()


# In[11]:


#removing the outliers (like the bookings that are made more than 2 year before the flight (illogic))
(df.purchase_lead > 600 ).value_counts()


# In[12]:


df[df.purchase_lead > 600]


# In[13]:


#having only the data of bookings before 600 days
df = df[df.purchase_lead < 600]


# In[14]:


#number of days between travel date and booking date

# Set the figure size
plt.figure(figsize=(15,5))

# Create the histogram with KDE
sns.histplot(data=df, x="purchase_lead", binwidth=20, kde=True)

# Set the x-axis label with units (days)
plt.xlabel("Purchase Lead Time", fontsize=10)

# Set the y-axis label with a description (number of people)
plt.ylabel("Bookings Count", fontsize=10)

# Show the plot
plt.show()


# ## Length of Stay

# In[15]:


plt.figure(figsize=(15,5))
sns.histplot(data=df, x="length_of_stay", binwidth=15,kde=True)


# In[16]:


(df.length_of_stay> 200).value_counts()


# In[17]:


df[df.length_of_stay> 500].booking_complete.value_counts()


# In[18]:


#filtering the data to have only length of stay days less than 500 days
df = df[df.purchase_lead <500 ]


# ## Flight Day

# In[19]:


mapping = {
    "Mon" : 1,
    "Tue" : 2,
    "Wed" : 3,
    "Thu" : 4,
    "Fri" : 5,
    "Sat" : 6,
    "Sun" : 7
}

df.flight_day = df.flight_day.map(mapping)


# In[20]:


df.flight_day.value_counts()


# ## Booking Origin

# In[21]:


plt.figure(figsize=(15,5))
ax = df.booking_origin.value_counts()[:20].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of bookings")


# In[22]:


plt.figure(figsize=(15,5))
ax = df[df.booking_complete ==1].booking_origin.value_counts()[:20].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of complete bookings")


# ## Booking Complete

# In[23]:


successful_booking_per = df.booking_complete.value_counts().values[0] / len(df) * 100


# In[24]:


unsuccessful_booking_per = 100-successful_booking_per


# In[25]:


print(f"Out of 50000 booking entries only {round(unsuccessful_booking_per,2)} % bookings were successfull or complete.")


# ### Export dataset to CSV

# In[27]:


df.to_csv("C:/Users/bendh/Desktop/data science/JN//filtered_customer_booking.csv")


# In[ ]:




