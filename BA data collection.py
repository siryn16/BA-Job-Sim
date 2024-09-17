#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


# In[2]:


#create an empty list to collect reviews
reviews = []
#create empty list to collect stars
stars = []
#create empty list to collect the date
date = []
#create empty list to collect the country the reviewer is from
country = []


# In[3]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 39
page_size = 100

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    #print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract review containers
    review_containers = soup.find_all("article", {"itemprop": "review"})

    for container in review_containers:
        # Extract review text
        review_text = container.find("div", {"class": "text_content"})
        if review_text:
            reviews.append(review_text.text.strip())
        else:
            reviews.append("None")

        # Extract star rating
        rating = container.find("div", {"class": "rating-10"})
        if rating:
            rating_value = rating.find("span", {"itemprop": "ratingValue"})
            if rating_value:
                stars.append(rating_value.text.strip())
            else:
                stars.append("None")
        else:
            stars.append("None")


        # Extract date
        review_date = container.find("time")
        if review_date:
            date.append(review_date.text.strip())
        else:
            date.append("None")

        # Extract country
        reviewer_info = container.find("h3", {"class": "text_sub_header userStatusWrapper"})
        if reviewer_info:
            country_text = reviewer_info.find("span").next_sibling
            if country_text:
                country.append(country_text.strip(" ()").strip())
            else:
                country.append("None")
        else:
            country.append("None")

    # Check if the number of reviews is less than the page size
    if len(review_containers) < page_size:
        print("Last page detected.")
        break  # Stop if the number of reviews is less than the expected page size
    if not review_containers:
        print("No reviews found, possibly end of pages.")
        break  # Stop if no reviews are found (end of data)





# In[4]:


len(reviews)


# In[5]:


len(stars)


# In[6]:


len(country)


# In[7]:


len(date)


# In[8]:


#create a DataFrame for these collected lists of data
df = pd.DataFrame({"reviews": reviews, "rating" : stars, "date" : date, "country" : country})


# In[9]:


df.head()


# In[10]:


df.shape


# In[13]:


df.to_csv("C:/Users/bendh/Desktop/data science/JN/BA_2.csv")


# In[ ]:




