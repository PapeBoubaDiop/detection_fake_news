from flask import Flask, render_template, request, redirect, url_for, flash
import os
import joblib
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete'

class FrequencyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=0.05, max_df=0.95):
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = CountVectorizer(min_df=self.min_df, max_df=self.max_df)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X_counts = self.vectorizer.transform(X)
        filtered = []
        vocab = self.vectorizer.get_feature_names_out()
        for row in X_counts:
            indices = row.nonzero()[1]
            words = [vocab[i] for i in indices]
            filtered.append(" ".join(words))
        return filtered

def preprocess_text(texts, language="english", stopwords=None, normalizer="stem"):
    valid_normalizer = (None, "stem", "lemma")
    if normalizer not in valid_normalizer:
        raise ValueError(f"normalizer must be in {valid_normalizer}")

    def process_one(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'(http|www)\S*', '', text)
        text = re.sub(r'\S*@\S*\s*', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        tokens = word_tokenize(text)
        if isinstance(stopwords, (set, list, tuple)):
            tokens = [w for w in tokens if w not in stopwords]
        if normalizer == "stem":
            stemmer = SnowballStemmer(language, ignore_stopwords=True)
            tokens = [stemmer.stem(w) for w in tokens]
        return " ".join(tokens)

    if isinstance(texts, pd.DataFrame):
        texts = texts.iloc[:, 0]

    return texts.apply(process_one)

# Chargement du pipeline multinomial Naive Bayes
model_path = os.path.join('models', 'pipeline_mnb.pkl')
try:
    model = joblib.load(model_path)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    model = None
    print(f"❌ Erreur lors du chargement du modèle : {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    result_color = None
    form_data = {
        'day': request.form.get('day', ''),
        'month': request.form.get('month', ''),
        'year': request.form.get('year', ''),
        'title': request.form.get('title', ''),
        'text': request.form.get('text', ''),
        'subject': request.form.get('subject', '')
    }

    if request.method == 'POST':
        try:
            if model is None:
                raise ValueError("Le modèle n'a pas été chargé correctement.")

            df = pd.DataFrame([{
                'text': form_data['text'],
                'title': form_data['title'],
                'subject': form_data['subject'],
                'day': int(form_data['day']),
                'month': int(form_data['month']),
                'year': int(form_data['year'])
            }])
            prediction = model.predict(df)[0]
            prediction_result = "Fake new" if prediction == 1 else "True new "
            result_color = "#e74c3c" if prediction == 1 else "#2ecc71"
        except Exception as e:
            flash(f"Erreur lors de la prédiction : {e}", 'danger')

    return render_template('prediction.html', prediction_result=prediction_result, result_color=result_color, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
