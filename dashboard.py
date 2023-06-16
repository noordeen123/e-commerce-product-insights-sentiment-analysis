import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the Excel file
data = pd.read_excel(f'iphone11.xlsx')

# Chart 1: Pie chart of sentiment distribution
sentiment_counts = data['predicted_sentiment'].value_counts()
plt.figure(figsize=(6, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.savefig('sentiment_pie_chart.png')
plt.close()

# Chart 2: Bar chart of average rating by product variation

#product_variation_rating = data.groupby('variation')['rating'].mean()
#plt.figure(figsize=(10, 6))
#product_variation_rating.plot(kind='bar')
#plt.xlabel('Product Variation')
#plt.ylabel('Average Rating')
#plt.title('Average Rating by Product Variation')
#plt.xticks(rotation=45)
#plt.savefig('rating_bar_chart.png')
#plt.close()

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
# Generate the HTML file
html_content = '''
<html>
<head>
<title>Product Analysis</title>
</head>
<body>
<h1>Product Analysis</h1>
<h2>Sentiment Distribution</h2>
<img src="sentiment_pie_chart.png" alt="Sentiment Distribution Pie Chart">

<h2>Average Rating by Product Variation</h2>
<img src="rating_bar_chart.png" alt="Average Rating by Product Variation">

<h2>Review Count Over Time</h2>
<img src="review_line_chart.png" alt="Review Count Over Time">

<h2>Top 10 Products by Count</h2>
<img src="product_name_bar_chart.png" alt="Top 10 Products by Count">

<h2>Top 10 Most Common Negative Reviews</h2>
<img src="negative_reviews_bar_chart.png" alt="Top 10 Most Common Negative Reviews">

<h2>Top 10 Most Common Positive Reviews</h2>
<img src="positive_reviews_bar_chart.png" alt="Top 10 Most Common Positive Reviews">
</body>
</html>
'''

# Save the HTML content to a file
with open('dashboard.html', 'w') as f:
    f.write(html_content)
