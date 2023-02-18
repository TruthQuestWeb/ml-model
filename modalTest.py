import modal
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

web_app = FastAPI()
image=(modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "apt-get update",
        "curl -O https://raw.githubusercontent.com/TruthQuestWeb/ml-model/main/train.csv",
    ).pip_install(
        "pandas",
        "scikit-learn",
        "numpy",
        "requests"
    ))

stub = modal.Stub(image=image)

class Article(BaseModel):
    text: str

@stub.function()
def foo(articles):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

    import pandas as pd
    df = pd.read_csv('/train.csv')

    import numpy as np
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=.2)

    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, y_train)

    df = pd.DataFrame({'text': [articles]})
    input = count_vectorizer.transform(df)
    pred = nb_classifier.predict(input)
    if pred == 1:
        return jsonable_encoder({'result': 'true'})
    else:
        return jsonable_encoder({'result': 'false'})

@web_app.post("/analysis/")
async def analysis(articleobj: Article):
    return foo.call(articleobj.text)

@stub.asgi(image=image)
def fastapi_App():
    return web_app

if __name__ == "__main__":
    stub.deploy("webapp")
