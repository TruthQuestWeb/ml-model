import modal

stub = modal.Stub(
    image=modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "apt-get update",
        "curl -O https://raw.githubusercontent.com/TruthQuestWeb/ml-model/main/train.csv",
    ).pip_install(
        "pandas",
        "numpy",
        "nltk",
        "scikit-learn",
    ),
)

@stub.function()
def foo():
    import re
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    nltk.download('stopwords')
    data = pd.read_csv('/train.csv')
    data = data.fillna('')
    #data['content'] = data['author'] + ' ' + data['title']
    X = data.drop(columns='label', axis=1)
    Y = data['label']

    port_stem = PorterStemmer()
    def stemming(content):
        review = re.sub('[^a-zA-Z]', ' ', content)
        review = review.lower()
        review = review.split()
        review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        return review

    data['text'] = data['text'].apply(stemming)

    X = data['text'].values
    Y = data['label'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X[0])
    if X_train_prediction[0] == 0:
        print("heck yes")
    else:
        print("no")

    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(X1_train, Y1_train)
    prediction1 = classifier.predict(X[0])
    if prediction1[0] == 0:
        print("hell yeah")
    else:
        print("no")

    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    from sklearn.linear_model import PassiveAggressiveClassifier
    linear_clf = PassiveAggressiveClassifier(max_iter=50)

    linear_clf.fit(X2_train, Y2_train)
    prediction2 = linear_clf.predict(X[0])
    if prediction2[0] == 0:
        print("shmyeah")
    else:
        print("nahhh ain't no way")

@stub.local_entrypoint
def main():
    author = "Toluse Olorunnipa"
    title = "Buttigieg, White House face backlash in aftermath of Ohio derailment"
    foo.call()
