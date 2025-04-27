import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
import sqlite3

def charger_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extraire_intentions(sql):
    select = re.search(r"SELECT\s+(.*?)\s+FROM", sql)
    where = re.search(r"WHERE\s+(.+)", sql)
    return select.group(1).strip() if select else '', where.group(1).strip() if where else ''

def obtenir_labels(sql):
    select, where = extraire_intentions(sql)
    select_labels = [label.strip() for label in select.replace(",", " ").split()]
    where_labels = [condition.strip() for condition in where.split("AND")]
    return select_labels, where_labels

def analyser_requete(requete, df):
    concepts = []
    champs = {
        'titre': set(df['titre']),
        'realisateur': set(df['realisateur']),
        'genre': set(df['genre']),
        'annee': set(map(str, df['annee'])),
        'acteur': set(df['acteur1']).union(df['acteur2'], df['acteur3'])
    }
    analyse = requete
    for champ, valeurs in champs.items():
        for val in valeurs:
            if val in requete:
                concepts.append((champ, val))
                analyse = analyse.replace(val, champ)
    return analyse, concepts

data_train = charger_json("queries_french_para.json")
data_eval = charger_json("queries_french_para_eval.json")
values_eval = charger_json("queries_french_para_eval_values.json")
films_df = pd.read_csv("base_films_500.csv")

X_train, y_select_train, y_where_train = [], [], []
for item in data_train:
    sql = item['sql']
    select, where = extraire_intentions(sql)
    phrases = [item['french']['query_french']] + item['french']['paraphrase_french']
    for phrase in phrases:
        X_train.append(phrase)
        y_select_train.append(select)
        y_where_train.append(where)

def entrainer_pipeline(X, y):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', Perceptron(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline

select_model = entrainer_pipeline(X_train, y_select_train)
where_model = entrainer_pipeline(X_train, y_where_train)

X_eval, y_select_eval, y_where_eval = [], [], []
for item in data_eval:
    req = item['french']['query_french']
    sel, whe = extraire_intentions(item['sql'])
    X_eval.append(req)
    y_select_eval.append(sel)
    y_where_eval.append(whe)

pred_select = select_model.predict(X_eval)
pred_where = where_model.predict(X_eval)

print("\n=== Rapport SELECT ===")
print(classification_report(y_select_eval, pred_select, zero_division=0))

print("\n=== Rapport WHERE ===")
print(classification_report(y_where_eval, pred_where, zero_division=0))

#print("\n=== Exemples de prédictions ===")
#for i in range(min(10, len(X_eval))):
    #print(f"\nQuestion : {X_eval[i]}\nSELECT attendu : {y_select_eval[i]}\nSELECT prédit  : {pred_select[i]}\nWHERE attendu : {y_where_eval[i]}\nWHERE prédit  : {pred_where[i]}")

def entrainer_pipeline_tfidf(X, y):
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', Perceptron(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline

select_model_tfidf = entrainer_pipeline_tfidf(X_train, y_select_train)
where_model_tfidf = entrainer_pipeline_tfidf(X_train, y_where_train)

pred_select_tfidf = select_model_tfidf.predict(X_eval)
pred_where_tfidf = where_model_tfidf.predict(X_eval)

accuracy_select = accuracy_score(y_select_eval, pred_select)
f1_select = f1_score(y_select_eval, pred_select, average='weighted', zero_division=0)

accuracy_where = accuracy_score(y_where_eval, pred_where)
f1_where = f1_score(y_where_eval, pred_where, average='weighted', zero_division=0)

accuracy_select_tfidf = accuracy_score(y_select_eval, pred_select_tfidf)
f1_select_tfidf = f1_score(y_select_eval, pred_select_tfidf, average='weighted', zero_division=0)

accuracy_where_tfidf = accuracy_score(y_where_eval, pred_where_tfidf)
f1_where_tfidf = f1_score(y_where_eval, pred_where_tfidf, average='weighted', zero_division=0)

summary_df = pd.DataFrame({
    "Tâche": ["SELECT", "WHERE"],
    "Accuracy (CountVectorizer)": [accuracy_select, accuracy_where],
    "F1 Score (CountVectorizer)": [f1_select, f1_where],
    "Accuracy (TF-IDF)": [accuracy_select_tfidf, accuracy_where_tfidf],
    "F1 Score (TF-IDF)": [f1_select_tfidf, f1_where_tfidf]
})
print("\n===Performances du modèle sur l'échantillon de test : ===")
print(summary_df)

with open("queries_french_para_eval_values.json", "r", encoding="utf-8") as f:
    valeur_test = json.load(f)

conn = sqlite3.connect(":memory:")
films_df.to_sql("films", conn, index=False, if_exists="replace")

nb_total = len(valeur_test)
nb_correct = 0
erreurs = []

for item in valeur_test:
    question = item["query"]
    valeur_attendue = item["value"]
    analyse, concepts_valeurs = analyser_requete(question, films_df)
    conditions = []
    for concept, valeur in concepts_valeurs:
        if concept == "acteur":
            cond = f"(acteur1 = '{valeur}' OR acteur2 = '{valeur}' OR acteur3 = '{valeur}')"
        else:
            cond = f"{concept} = '{valeur}'"
        conditions.append(cond)

    where_clause = " AND ".join(conditions)

    if "realisateur" in question.lower():
        select_clause = "realisateur"
    elif "année" in question.lower() or "date" in question.lower():
        select_clause = "annee"
    elif "titre" in question.lower() or "film" in question.lower():
        select_clause = "titre"
    else:
        select_clause = "*" 

    requete_sql = f"SELECT {select_clause} FROM films WHERE {where_clause} LIMIT 1;"

    try:
        curseur = conn.execute(requete_sql)
        resultat = curseur.fetchone()
        resultat_str = str(resultat[0]) if resultat else None
    except Exception as e:
        resultat_str = None
        erreurs.append({
            "question": question,
            "sql": requete_sql,
            "error": str(e),
            "expected": valeur_attendue,
            "predicted": resultat_str
        })
        continue

    if resultat_str == valeur_attendue:
        nb_correct += 1
    else:
        erreurs.append({
            "question": question,
            "sql": requete_sql,
            "expected": valeur_attendue,
            "predicted": resultat_str
        })

accuracy = nb_correct / nb_total

print("\n=== Erreurs de résultats : ===")
print(pd.DataFrame(erreurs))
print(f"\nAccurary : {accuracy} ")

def interface_interactive(select_model, where_model, df_films):
    print("\nInterface interactive de requêtes en langage naturel (tapez 'exit' pour quitter) :")
    while True:
        phrase = input("\nEntrez votre requête : ")
        if phrase.lower() == 'exit':
            break

        select_pred = select_model.predict([phrase])[0]
        where_pred = where_model.predict([phrase])[0]

        _, concepts_valeurs = analyser_requete(phrase, df_films)
        print("\nValeurs détectées :", concepts_valeurs)

        where_pred = where_pred.rstrip(';')
        print("\nRequête SQL prédite :")
        print(f"SELECT {select_pred} FROM films WHERE {where_pred};")

interface_interactive(select_model, where_model, films_df)