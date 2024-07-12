import joblib
import tkinter as tk
from tkinter import scrolledtext
def classify_email(input_text, vectorizer, model):
    input_text = [input_text]  # Transform input into a list (single email)
    input_tfidf = vectorizer.transform(input_text)
    prediction = model.predict(input_tfidf)
    return "Spam" if prediction == 1 else "Safe"

def classify_from_ui():
    input_text = text_area.get("1.0", tk.END)  # Get text from the Tkinter text area
    result = classify_email(input_text, vectorizer, model)
    result_label.config(text=f"Classification result: {result}")

if __name__ == "__main__":
    # Load the saved model and vectorizer
    model = joblib.load('svc_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    window = tk.Tk()
    window.title("Email Classifier")

    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10, font=("Arial", 12))
    text_area.pack(pady=10)

    classify_button = tk.Button(window, text="Classify Email", width=15, height=2, command=classify_from_ui)
    classify_button.pack(pady=10)

    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.pack(pady=10)

    window.mainloop()
