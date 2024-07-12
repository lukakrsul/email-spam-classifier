import os
import email
import numpy as np
from email.policy import default

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bs4 import BeautifulSoup
from sklearn.svm import SVC



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
    return ""  #In case no plain text or HTML is found



#Set the paths to your folders
pwd = os.getcwd()
spam_folder = os.path.join(pwd, 'spam')
ham_folder = os.path.join(pwd, 'ham')

#Load the data
spam_emails, ham_emails = load_spam_and_nonspam_data(spam_folder, ham_folder)

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_content = []
for email in X_train:
    X_train_content.append(email_to_text(email))

X_test_content = []
for email in X_test:
    X_test_content.append(email_to_text(email))

#Vectorizing the data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_content)
X_test_tfidf = vectorizer.transform(X_test_content)

model = SVC()
model.fit(X_train_tfidf, y_train)

prediction = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)
report = classification_report(y_test, prediction)

print(report)

print(len(ham_emails))
print(len(spam_emails))