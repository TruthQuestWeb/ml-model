import requests
import os
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time

from youdotcom import Search

def extract_source(url):
    start = url.find("www.") + 4
    end = url.find(".com")
    source = url[start:end]
    return source

def search_youdotcom(url):
    article_response = requests.get(url)
    soup = BeautifulSoup(article_response.text, "html.parser")
    article = soup.get_text(" ", strip=True)

    title = soup.find("title").text
    search_results = Search.search_for(title)

    parsed = json.loads(search_results['results'])
    print(json.dumps(parsed, indent=4))


def search_initial_article(url):
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    ENGINE_ID = os.getenv("ENGINE_ID")
    
    # Get the article text and title from the URL
    article_response = requests.get(url)
    soup = BeautifulSoup(article_response.text, "html.parser")
    article = soup.get_text(" ", strip=True)

    title = soup.find("title").text

    
    # Search the web for similar articles using the article title
    response = requests.get("https://www.googleapis.com/customsearch/v1?q=" + title + "&cx=" + ENGINE_ID + "&key=" + API_KEY)
    results = response.json()["items"]
    

    comparison_articles = {}
    comparison_articles[url] = article

    sources = []
    sources.append(extract_source(url))
    
    for result in results[:5]:
        if extract_source(result["link"]) not in sources:
            sources.append(extract_source(result["link"]))
            article_response = requests.get(result["link"])
            soup = BeautifulSoup(article_response.text, "html.parser")
            article = soup.get_text(" ", strip=True)
            comparison_articles[result["link"]] = article

            print("Comparison article:", result["link"])


    print(json.dumps(comparison_articles, indent=4))

    return comparison_articles