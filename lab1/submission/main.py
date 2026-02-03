import pandas as pd

from src.NaiveClassifier import NaiveSpamClf


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def process_features(train, test):
    # Converting text features (messages) into numbers
    feature_extraction = TfidfVectorizer()
    train = feature_extraction.fit_transform(train)
    test = feature_extraction.transform(test)    

    return train, test



def evaluate_model(X_train, X_test, y_train, y_test, model):
    # Fit x data(features) and y data (labels)
    model.fit(X_train, y_train)
    
    # Get the predictions on unseen test data 
    pred_on_test = model.predict(X_test)

    # Print classification report
    print(classification_report(y_pred=pred_on_test, y_true=y_test))


def main():
    # We start by reading the given data set into the program
    df = pd.read_csv('data/processed_data.csv')

    # Seperate features and labels 
    features = df.drop(columns=['label'])
    label = df['label']

    # Creating this for future use by NaiveSpamClf()
    X_texts = (
            features['subject'].fillna('') + "\n" + features['message'].fillna('')
    )


    # Naive classifier
    nclf = NaiveSpamClf()
    
    # Logistic regression classifier
    lr = LogisticRegression()

    # Multinomial Naive Baye's classifier
    mb = MultinomialNB()

    # for classification with naive classifier text is required
    # evaluation of naive model
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)
    
    # Naive model predictions and evaluation
    print("Evaluating Naive classifier")
    evaluate_model(X_train, X_test, y_train, y_test, nclf)
    print("Naive classifier done\n")

    # process text for logistic regression and multinomial nb
    
    X_train, X_test, y_train, y_test = train_test_split(X_texts, label, test_size=0.3, random_state=42)

    print("Vectorizing features")
    vectorized_text = process_features(X_train, X_test)
    X_train = vectorized_text[0]
    X_test = vectorized_text[1]

    # Multinomial NB
    print("Evaluating Multinomial NB")
    evaluate_model(X_train, X_test, y_train, y_test, mb)
    print("Multinomial NB done\n")

    
    # logistic regression
    print("Evaluating Logistic regression")
    evaluate_model(X_train, X_test, y_train, y_test, lr)
    print("Logistic regression done\n")

main()