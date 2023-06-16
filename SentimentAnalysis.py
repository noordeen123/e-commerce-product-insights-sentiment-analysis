import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import csv
import os
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
import webbrowser
#import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')


# Function to preprocess a review text
def preprocess_text(text):
    # Check if the text is not NaN
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize into words
        words = word_tokenize(text)
        # Remove stop words
        words = [word for word in words if word not in stopwords.words('english')]
        # Join the words back into a string
        text = ' '.join(words)
    else:
        text = ''
    return text

# Function to classify a review sentiment using a sentiment dictionary and rating
def classify_sentiment(text, rating, sentiment_dict):
    # Create an instance of the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Get the sentiment scores for the text
    sentiment_scores = sid.polarity_scores(text)
    # Determine the sentiment label based on the compound score and rating
    if sentiment_scores['compound'] >= 0 and rating >= 3:
        return 'positive'
    else:
        return 'negative'

# Function to perform topic modeling using LDA
#def perform_topic_modeling(texts, num_topics):
#   vectorizer = TfidfVectorizer()
#   X = vectorizer.fit_transform(texts)
#   lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#   lda.fit(X)
#   return lda, X

# Function to classify reviews using SVM model and sentiment dictionary
def classify_reviews(filename, sentiment_dict, num_topics):
    # Load reviews into a pandas dataframe
    reviews_df = pd.read_csv(filename)

    # Preprocess the review texts
    reviews_df['text'] = reviews_df['text'].apply(preprocess_text)

    # Perform topic modeling on the preprocessed texts
#   lda, X = perform_topic_modeling(reviews_df['text'], num_topics)

    # Transform the preprocessed texts using the LDA model
#    X_topics = lda.transform(X)

    # Create a TF-IDF vectorizer and fit it on the preprocessed texts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews_df['text'])

    # Train an SVM model on the vectorized texts and the sentiment scores
    clf = svm.SVC(kernel='linear')
    #clf.fit(X, reviews_df['sentiment'])

    # Classify the reviews using the trained SVM model and the sentiment dictionary
    reviews_df['predicted_sentiment'] = reviews_df.apply(lambda x: classify_sentiment(x['text'], x['rating'], sentiment_dict), axis=1)

    # Add topic probabilities to the dataframe
#   topic_columns = ['topic_{}'.format(i) for i in range(num_topics)]
#   for i, column in enumerate(topic_columns):
#       reviews_df[column] = X_topics[:, i]

    # Save the classified reviews to a new CSV file
    output_filename = os.path.splitext(filename)[0] + '.csv'
    reviews_df.to_csv(output_filename, index=False)

    return output_filename

# Load the sentiment dictionary
sentiment_dict = {}
with open('dataset/dictionary.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        sentiment_dict[row[0]] = int(row[1])


# Classify the reviews in the CSV file with 5 topics in the LDA model
#product_name = st.text_input("Enter the product name:")
product_name = input("ENTER THE PRODUCT NAME: ")
reviews_filename = f'dataset/{product_name}.csv'
num_topics = 5
classified_file = classify_reviews(reviews_filename, sentiment_dict, num_topics)

# Load the classified reviews into a pandas dataframe
classified_df = pd.read_csv(classified_file)

# Save the classified reviews to an Excel file with the same name as the input file
output_filename = os.path.splitext(classified_file)[0] + '.xlsx'
classified_df.to_excel(output_filename, index=False)

wb = pd.read_excel(f'dataset/{product_name}.xlsx') # This reads in your excel doc as a pandas DataFrame

wb.to_html(f'{product_name}.html') # Export the DataFrame (Excel doc) to an html file 

import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the Excel file
data = pd.read_excel(f'dataset/{product_name}.xlsx')

# Chart 1: Pie chart of sentiment distribution
sentiment_counts = data['predicted_sentiment'].value_counts()
plt.figure(figsize=(6, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.savefig('sentiment_pie_chart.png')
plt.close()

# Chart 2: Bar chart of average rating by product variation

product_variation_rating = data.groupby('variation')['rating'].mean()
plt.figure(figsize=(10, 6))
product_variation_rating.plot(kind='bar')
plt.xlabel('Product Variation')
plt.ylabel('Average Rating')
plt.title('Average Rating by Product Variation')
plt.xticks(rotation=45)
plt.savefig('rating_bar_chart.png')
plt.close()

# Chart 3: Line chart of review count over time
data['date'] = pd.to_datetime(data['date'], format='mixed')
review_count_over_time = data.groupby(data['date'].dt.month)['text'].count()
plt.figure(figsize=(10, 6))
review_count_over_time.plot(kind='line', marker='o')
plt.xlabel('Month')
plt.ylabel('Review Count')
plt.title('Review Count Over Time')
plt.xticks(range(1, 13))
plt.savefig('review_line_chart.png')
plt.close()

# Chart 4: Bar chart of count by product name
if not data.empty:
    product_name_counts = data['App_Name'].value_counts().head(10)
    if not product_name_counts.empty:
        plt.figure(figsize=(10, 6))
        product_name_counts.plot(kind='barh')
        plt.ylabel('Product Name')
        plt.xlabel('Count')
        plt.title('Top 10 Products by Count')
        plt.savefig('product_name_bar_chart.png')
        plt.close()

# Chart 5: Bar chart of most common negative reviews
negative_reviews = data[data['predicted_sentiment'] == 'negative']
if not negative_reviews.empty:
    negative_reviews_count = negative_reviews['text'].value_counts().head(5)
    if not negative_reviews_count.empty:
        plt.figure(figsize=(10, 6))
        negative_reviews_count.plot(kind='barh')
        plt.xlabel('Count')
        plt.ylabel('Review')
        plt.title('Top 5 Most Common Negative Reviews')
        plt.savefig('negative_reviews_bar_chart.png')
        plt.close()

# Chart 6: Bar chart of most common positive reviews
positive_reviews = data[data['predicted_sentiment'] == 'positive']
if not positive_reviews.empty:
    positive_reviews_count = positive_reviews['text'].value_counts().head(10)
    if not positive_reviews_count.empty:
        plt.figure(figsize=(10, 6))
        positive_reviews_count.plot(kind='barh')
        plt.xlabel('Count')
        plt.ylabel('Review')
        plt.title('Top 10 Most Common Positive Reviews')
        plt.savefig('positive_reviews_bar_chart.png')
        plt.close()

# Generate the HTML file
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Product Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-top: 50px;
            text-transform: uppercase;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #999;
            padding-bottom: 10px;
            text-align: center;
            transition: color 0.3s ease;
        }
        .chart-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            text-align: center;
            animation: fadeIn 1s;
            transition: background-color 0.3s ease;
        }
        .chart-container:hover {
            background-color: #f9f9f9;
        }
        .chart-container:hover h2 {
            color: #333;
        }
        img {
            max-width: 100%;
            border: 1px solid #ccc;
            display: block;
            margin: 0 auto;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            animation: zoomIn 1s;
            transition: transform 0.3s ease;
        }
        img:hover {
            transform: scale(1.05);
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        @keyframes zoomIn {
            from {
                transform: scale(0);
            }
            to {
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Product Analysis</h1>
    </header>
    <main>
        <section class="chart-container" style="background-color: #f9f9f9;">
            <h2 class="chart-title">Sentiment Distribution</h2>
            <img src="sentiment_pie_chart.png" alt="Sentiment Distribution Pie Chart">
        </section>

        <section class="chart-container" style="background-color: #f7f7f7;">
            <h2 class="chart-title">Average Rating by Product Variation</h2>
            <img src="rating_bar_chart.png" alt="Average Rating by Product Variation">
        </section>

        <section class="chart-container" style="background-color: #f5f5f5;">
            <h2 class="chart-title">Review Count Over Time</h2>
            <img src="review_line_chart.png" alt="Review Count Over Time">
        </section>

        <section class="chart-container" style="background-color: #f3f3f3;">
            <h2 class="chart-title">Top 10 Products by Count</h2>
            <img src="product_name_bar_chart.png" alt="Top 10 Products by Count">
        </section>

        <section class="chart-container" style="background-color: #f1f1f1;">
            <h2 class="chart-title">Top 10 Most Common Negative Reviews</h2>
            <img src="negative_reviews_bar_chart.png" alt="Top 10 Most Common Negative Reviews">
        </section>

        <section class="chart-container" style="background-color: #efefef;">
            <h2 class="chart-title">Top 10 Most Common Positive Reviews</h2>
            <img src="positive_reviews_bar_chart.png" alt="Top 10 Most Common Positive Reviews">
        </section>
    </main>
    <footer>
        <p>&copy; 2023 Product Analysis. All rights reserved.</p>
    </footer>
</body>
</html>
'''

# Save the HTML content to a file
with open('dashboard.html', 'w') as f:
    f.write(html_content)
#    html_content = wb.to_html()
#st.write(html_content, unsafe_allow_html=True)

# Display the HTML file
#webbrowser.open_new_tab('dashboard.html')
webbrowser.Chrome.open_new_tab('dashboard.html')
#webbrowser.open_new_tab(f'{product_name}.html')
