import os
import email
import numpy as np
from email.policy import default
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from bs4 import BeautifulSoup
import joblib

def load_emails_from_folder(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='latin-1') as file:
            msg = email.message_from_file(file, policy=default)
            emails.append(msg)
    return emails

def load_spam_and_nonspam_data(spam_folder, nonspam_folder):
    spam_emails = load_emails_from_folder(spam_folder)
    nonspam_emails = load_emails_from_folder(nonspam_folder)
    return spam_emails, nonspam_emails

def html_to_plain_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
    return ""

def train_model_and_save(spam_folder, ham_folder, model_output_file, vectorizer_output_file):
    # Load the data
    spam_emails, ham_emails = load_spam_and_nonspam_data(spam_folder, ham_folder)

    X = np.array(ham_emails + spam_emails, dtype=object)
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_content = [email_to_text(email) for email in X_train]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_content)

    model = SVC()
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer
    joblib.dump(model, model_output_file)
    joblib.dump(vectorizer, vectorizer_output_file)

    print(f"Model and vectorizer saved to {model_output_file} and {vectorizer_output_file}")

if __name__ == "__main__":
    pwd = os.getcwd()
    spam_folder = os.path.join(pwd, 'spam')
    ham_folder = os.path.join(pwd, 'ham')

    # Define output file paths
    model_output_file = 'svc_model.joblib'
    vectorizer_output_file = 'tfidf_vectorizer.joblib'

    # Train the model and save
    train_model_and_save(spam_folder, ham_folder, model_output_file, vectorizer_output_file)


