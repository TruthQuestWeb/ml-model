#Bring in search.py
from search import search_initial_article, search_youdotcom
from analysis import initialize_data

# Example usage
url = "https://www.theonion.com/loyal-dog-spends-hours-each-day-humping-owner-s-grave-1850042397"

#search_youdotcom(url)

articles = search_initial_article(url)
print(articles)
#print("The article is classified as:", result, "with an average similarity score of", average)

df = initialize_data(articles)
print(df)

df.to_csv('results.csv', index=False)
