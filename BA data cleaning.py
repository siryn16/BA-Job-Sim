#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[27]:


df = pd.read_csv("C:/Users/bendh/Desktop/data science/JN/BA_2.csv", index_col=0)


# In[28]:


df.head()


# In[123]:


df.tail()


# In[29]:


df.dtypes


# In[30]:


df.isna().sum()


# In[31]:


df[df['rating'].isna() == 1]


# In[32]:


#refilling na ratings based on total rating in the website


# In[33]:


df.loc[3276, 'rating'] = 5


# In[35]:


print(df.loc[3276])


# In[36]:


df.loc[3409, 'rating'] = 6


# In[37]:


df.loc[3433, 'rating'] = 5


# In[38]:


df.loc[3662, 'rating'] = 1


# In[39]:


df.loc[3696, 'rating'] = 3


# In[40]:


df[df['rating'].isna() == 1]


# In[41]:


#refilling na countries based on the review text in the website


# In[42]:


df[df['country'].isna() == 1]


# In[43]:


df.loc[3519, 'country'] = 'Saint Lucia'


# In[44]:


df.loc[3215, 'country'] = 'United Kingdom'


# In[45]:


df[df['country'].isna() == 1]


# In[46]:


df.isna().sum()


# In[47]:


#converting rating from float to integers


# In[48]:


df['rating'] = df['rating'].astype(int)


# In[49]:


df.dtypes


# In[50]:


df.head()


# In[51]:


#convert the dates to a date format


# In[52]:


from datetime import datetime
import re


# In[53]:


# Function to clean and convert date
def convert_date(date_string):
    date_string_clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_string)
    return pd.to_datetime(date_string_clean, format='%d %B %Y')


# In[54]:


# Apply function to the DataFrame column
df['date'] = df['date'].apply(convert_date)


# In[55]:


df.head()


# In[56]:


df['date'] = df['date'].dt.strftime('%d-%m-%Y')


# In[57]:


df.head()


# In[58]:


#cleaning reviews


# In[59]:


df.dtypes


# In[60]:


def clean_review(review):
    if pd.isna(review):
        return review
    # Define patterns to remove
    patterns = [r'Not Verified', r'✅ Trip Verified',r'Verified Review', r'|']
    
    # Remove patterns using regular expressions
    for pattern in patterns:
        review = re.sub(pattern, '', review)
    
    # Remove extra spaces that may result from removal
    review = ' '.join(review.split())
    return review


# In[61]:


df.head()


# In[62]:


df['reviews'] = df['reviews'].apply(clean_review)


# In[63]:


df.head()


# In[64]:


df['reviews'] = df['reviews'].str.strip('| ').str.lstrip()


# In[65]:


reviews_data = df['reviews']


# In[66]:


df.head()


# In[67]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[43]:


nltk.download('stopwords')
nltk.download('wordnet')


# In[44]:


nltk.download('vader_lexicon')


# In[68]:


#loop through each review : remove punctuations, small case it , join it and add it to corp
# Initialize lemmatizer and stopwords
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Initialize the corpus list
corpus = []

# Loop through each review
for rev in reviews_data:
    # Remove punctuation and keep only alphabetic characters
    rev = re.sub('[^a-zA-Z]', ' ', rev)
    
    # Convert all characters to lowercase
    rev = rev.lower()
    
    # Split the review into a list of words
    rev = rev.split()
    
    # Lemmatize each word and remove stopwords
    rev = [lemma.lemmatize(word) for word in rev if word not in stop_words]
    
    # Join the processed words back into a single string
    rev = " ".join(rev)
    
    # Add the processed review to the corpus
    corpus.append(rev)

# Ensure corpus matches the length of reviews_data
if len(corpus) != len(reviews_data):
    raise ValueError(f"Length mismatch: corpus length is {len(corpus)}, but reviews_data length is {len(reviews_data)}")

# Add the corpus to the DataFrame
df['corpus'] = corpus


# In[69]:


print(len(corpus))  # Should match the number of rows in df
print(len(df['reviews'])) 


# In[70]:


df['corpus'] = corpus


# In[71]:


print(df.loc[0, 'corpus'])


# In[72]:


print(df.loc[0, 'reviews'])


# In[73]:


#see which rating is the most and which one is the least


# In[74]:


df['rating'].value_counts()


# In[75]:


counted_values = df['rating'].value_counts().sort_index()
ax = df['rating'].value_counts().sort_index() \
                .plot(kind= 'bar',
                      title= 'Rating by stars',
                      figsize=(10,5))
ax.set_xlabel('Stars')
ax.set_ylabel('Score')
# Annotate each bar with the count value
for index, value in enumerate(counted_values):
    ax.text(index, value + 0.1, str(value), ha='center')
plt.show()


# In[76]:


# Calculate the average rating
average_rating = df['rating'].mean()

print(f"The average rating is: {average_rating:.2f}")


# In[77]:


# Get unique countries
unique_countries = df['country'].unique()

# Count the number of unique countries
unique_countries_count = len(unique_countries)

print(f"The number of countries is: {unique_countries_count}")


# In[78]:


# Calculate the sum of the ratings
total_reviews = df['reviews'].count()

print(f"The total number of reviews is: {total_reviews}")


# In[79]:


df.shape


# In[80]:


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# In[81]:


# Create lists to store sentiment scores
compound_scores = []
positive_scores = []
neutral_scores = []
negative_scores = []


# In[82]:


# Loop through the processed reviews
for review in df['corpus']:
    # Get the sentiment scores
    sentiment = analyzer.polarity_scores(review)
    
    # Append the scores to their respective lists
    compound_scores.append(sentiment['compound'])
    positive_scores.append(sentiment['pos'])
    neutral_scores.append(sentiment['neu'])
    negative_scores.append(sentiment['neg'])


# In[83]:


# Add the scores to the DataFrame
df['compound'] = compound_scores
df['positive'] = positive_scores
df['neutral'] = neutral_scores
df['negative'] = negative_scores


# In[84]:


def categorize_sentiment(compound_score, rating):
    # First, determine sentiment based on compound score
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
   # Adjust sentiment based on the rating
    if rating in [4, 5]:
        sentiment = 'Neutral'
    elif rating <= 3 and sentiment in ['Neutral', 'Positive']:
        sentiment = 'Negative'
    return sentiment


# In[85]:


# Apply the function to categorize sentiment
df['sentiment_category'] = df.apply(lambda row: categorize_sentiment(row['compound'], row['rating']), axis=1)


# In[86]:


# Count the number of each sentiment category
sentiment_counts = df['sentiment_category'].value_counts()


# In[87]:


sentiment_counts


# In[88]:


# Define a custom color palette
custom_palette = {
    'Positive': '#77DD77',  # Pastel green
    'Neutral': '#5F9BCC',   # Pastel yellow
    'Negative': '#FF6961'   # Pastel red
}

# Plot the sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=custom_palette)
plt.title('Sentiment Distribution of British Airways Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()


# In[89]:


# Show sentiment analysis results for the first review
first_review = df.loc[0, 'reviews']
first_sentiment = df.loc[0, ['compound', 'positive', 'neutral', 'negative', 'sentiment_category']]
print(f"Review: {first_review}\n")
print(f"Sentiment Analysis:\n{first_sentiment}")


# In[94]:


from nltk.tokenize import word_tokenize
nltk.download('punkt')


# In[95]:


from wordcloud import WordCloud


# In[96]:


# Initialize a dictionary to store all negative words and their cumulative scores
all_negative_words = {}


# In[97]:


# Loop through each review in the corpus
for review in df['corpus']:
    # Tokenize the review
    review_tokens = word_tokenize(review)
    
    # Loop through each word in the review
    for word in review_tokens:
        # Get the sentiment score for the word
        sentiment = analyzer.polarity_scores(word)
        
        # If the word has a negative sentiment, add it to the dictionary
        if sentiment['neg'] > 0:
            if word in all_negative_words:
                all_negative_words[word] += sentiment['neg']
            else:
                all_negative_words[word] = sentiment['neg']


# In[98]:


# Print the number of unique negative words found
print(f"Total unique negative words found: {len(all_negative_words)}")


# In[99]:


import matplotlib.colors as mcolors
# Define a custom colormap for red shades
red_colors = ["#660000", "#cc0000", "#ff4d4d", "#ff9999"]  # Dark red to lighter red
custom_red_colormap = mcolors.LinearSegmentedColormap.from_list('custom_red', red_colors, N=256)
# Generate a word cloud for all negative words
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=custom_red_colormap).generate_from_frequencies(all_negative_words)


# In[100]:


# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Negative Words in All Reviews')
plt.show()


# In[101]:


# Initialize a dictionary to store all negative words and their cumulative scores
all_positive_words = {}


# In[102]:


# Loop through each review in the corpus
for review in df['corpus']:
    # Tokenize the review
    review_tokens = word_tokenize(review)
    
    # Loop through each word in the review
    for word in review_tokens:
        # Get the sentiment score for the word
        sentiment = analyzer.polarity_scores(word)
        
        # If the word has a positive sentiment, add it to the dictionary
        if sentiment['pos'] > 0:
            if word in all_positive_words:
                all_positive_words[word] += sentiment['pos']
            else:
                all_positive_words[word] = sentiment['pos']


# In[103]:


# Print the number of unique positive words found
print(f"Total unique positive words found: {len(all_positive_words)}")


# In[104]:


# Define a custom colormap
colors = ["#004d00", "#006400", "#009900", "#66ff66"]  # Dark green to light green
custom_colormap = mcolors.LinearSegmentedColormap.from_list('custom_green', colors, N=256)
# Generate a word cloud for all positive words
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=custom_colormap).generate_from_frequencies(all_positive_words)


# In[105]:


# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Positive Words in All Reviews')
plt.show()


# In[106]:


# Combine all reviews into a single text
all_reviews = ' '.join(df['reviews'])
stopwords_set = set(word.lower() for word in stopwords.words('english'))
stopwords_set .update(['ba',"british","airway", "airline", "london", "heathrow", "plane", "choice", "even", "airline","aircraft", "took" \
                  "able" , "ok" , "really", "never", "Airways",".hour", "hour", "minute", "get", "one", "would", "Gatwick", "also", "back", "way"\
                      "flew" , "made" , "like" , "airport", "could", "despite", "given", "although", "told", "u", "airlines", "seem", "got", "quite" \
                      "used", "flew", "wife", "able", "say", "next", "take", "bit", "via" , "minutes", "still", "need", "way", "lhr", "make"\
                      "called", "felt", "rather", "since", "ask", "use", "left", "flight", "new", "york", "put", "jfk" ])


# In[107]:


# Define a custom colormap for blue shades
blue_colors = ["#003366", "#004080", "#0066cc", "#66b3ff"]  # Dark blue to lighter blue
custom_blue_colormap = mcolors.LinearSegmentedColormap.from_list('custom_blue', blue_colors, N=256)
# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=custom_blue_colormap, stopwords = stopwords_set ).generate(all_reviews)


# In[108]:


# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis
plt.title('WordCloud of All Words in Reviews')
plt.show()


# In[109]:


import nltk.collocations as collocations
from nltk import FreqDist, bigrams


# In[110]:


reviews = " ".join(df.corpus)
# Clean the text: remove special characters and make everything lowercase
reviews_cleaned = re.sub(r'[^\w\s]', '', reviews.lower())
#split the text of all reviews into a list of words
# Convert all words to lowercase before filtering
words = reviews_cleaned.lower().split(" ")

new_words = [word for word in words if word not in stopwords_set]

def get_freq_dist(new_words,number_of_ngrams):
    from nltk import ngrams
    ## generate bigrams
    ngrams = ngrams(new_words, number_of_ngrams)
    #creating FreqDist
    ngram_fd = FreqDist(ngrams).most_common(40)
    #sort values bu highest frequency
    ngram_sorted = {k:v for k,v in sorted(ngram_fd, key=lambda item:item[1])}
    #join bigram tokens with '_' + maintain sorting
    ngram_joined = {'_'.join(k):v for k,v in sorted (ngram_fd, key=lambda item:item[1])}
    #convert to pandas series for easy plotting
    ngram_freqdist = pd.Series(ngram_joined)
    plt.figure(figsize=(10,10))
    ax = ngram_freqdist.plot(kind="barh")
    return ax

get_freq_dist(new_words,4)


# In[112]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[113]:


# Function to get compound sentiment score
def get_compound_score(text):
    return analyzer.polarity_scores(text)['compound']

# Apply the function to get compound scores
df['compound'] = df['reviews'].apply(get_compound_score)


# In[115]:


# Plot the barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='rating', y='compound', hue='rating', palette='viridis')
ax.set_title('Compound Score by British Airways Review Stars')
ax.set_xlabel('Review Score')
ax.set_ylabel('Compound Score')
plt.legend(title='rating')
plt.show()


# In[116]:


# Plot the barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='rating', y='positive', hue='rating', palette='viridis')
ax.set_title('Compound Score by British Airways Review Stars')
ax.set_xlabel('Review Score')
ax.set_ylabel('Compound Score')
plt.legend(title='rating')
plt.show()


# In[117]:


print(df.columns)


# In[118]:


# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')


# In[119]:


df.head()


# In[120]:


# Extract year and month for aggregation
df['year_month'] = df['date'].dt.to_period('M')


# In[121]:


# Aggregate average sentiment score by year and month
monthly_sentiment = df.groupby('year_month')['compound'].mean()

# Convert PeriodIndex to Timestamp for plotting
monthly_sentiment.index = monthly_sentiment.index.to_timestamp()


# In[122]:


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(monthly_sentiment.index, monthly_sentiment.values, marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.title('Evolution of Review Sentiments Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[93]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download


# In[94]:


# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]


# In[95]:


# Apply preprocessing to the reviews
df['processed_reviews'] = df['reviews'].apply(preprocess)


# In[96]:


from gensim import corpora

# Create a dictionary and corpus
dictionary = corpora.Dictionary(df['processed_reviews'])
corpus = [dictionary.doc2bow(doc) for doc in df['processed_reviews']]


# In[97]:


from gensim import models

# Apply LDA
num_topics = 5  # You can adjust the number of topics based on your dataset
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Display topics
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)


# In[98]:


import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Visualize the topics
lda_viz = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_viz)


# In[ ]:




