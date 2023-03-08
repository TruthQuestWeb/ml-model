from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def initialize_data(articles_dict):
    articles = [{'url': url, 'article': article} for url, article in articles_dict.items()]
    df = pd.DataFrame(articles)

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(df['article'])

    similarity = cosine_similarity(features)
    print(similarity)

    model = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(features)
    

    df['cluster'] = model.predict(features)

    return df
