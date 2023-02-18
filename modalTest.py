import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(
        "pandas",
        "numpy",
        "nltk",
        "scikit-learn",
    ),
)

@stub.function
def foo():
    import pandas as pd
    import nltk
    import sklearn
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    data = pd.read_csv('https://github.com/TruthQuestWeb/treehacks/blob/main/train.csv')
    data = data.fillna('')
    data.content = data.author + ' ' + data.title
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

    data.content = data.content.apply(stemming)

    X = data.content.values
    Y = data.label.values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score of the training data : ', training_data_accuracy)

@stub.local_entrypoint
def main():
    foo.call()
