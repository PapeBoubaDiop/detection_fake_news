# 📰 Fake News Detection - NLP Project with Flask

Ce projet est une application web construite avec **Flask** pour détecter si un article est une **fake news** ou non. Le modèle utilisé est un pipeline de **Naive Bayes** incluant toutes les étapes de prétraitement du texte, d'encodage et de vectorisation.


---

## 📁 Structure du projet

fake-news-detection/  
│  
├── app.py  
├── models/  
│   └── pipeline_mnb.pkl  
├── templates/  
│   ├── base.html  
│   ├── home.html  
│   ├── about.html  
│   └── prediction.html  
├── static/  
│   ├── style.css  
│   └── img/  
│       └── logo.png  
├── fake_news_detection.ipynb  
└── README.md  

---

## ⚙️ Installation

```bash
git clone https://github.com/PapeBoubaDiop/detection_fake_news.git
cd detection_fake_news/deploiement

```

> 💡 Assure-toi d’utiliser une version compatible de `scikit-learn` (v1.6.1 recommandée) pour éviter les erreurs de désérialisation.

---

## 🚀 Lancer l'application

```bash
python app.py
```

L'application sera accessible à l’adresse : http://127.0.0.1:5000

---

## 🧠 Modèle utilisé

- **Modèle :** `MultinomialNB` dans un `Pipeline`
- **Colonnes utilisées :** `text`, `title`, `subject`, `day`, `month`, `year`
- **Prépocessing intégré** : nettoyage du texte, vectorisation TF-IDF, encodage

---

## 🧪 Mode d'emploi

1. Aller dans l’onglet **Prédiction**
2. Saisir :
   - Le titre de l’article
   - Le contenu de l’article
   - Le sujet
   - La date (jour, mois, année)
3. Cliquer sur **Prédire**
4. Résultat :
   - True new
   - Fake new

---


## ✍️ Auteurs

- Camara Sanor
- Diallo Mamadou Saïdou
- Diop Papa Boubacar
- Lo Ibrahima Fa

