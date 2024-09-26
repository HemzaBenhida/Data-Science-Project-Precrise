#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:07:31 2024

@author: hamzadriss
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

file_path = '/Users/hamzadriss/Downloads/PreCrise/heart_2022_with_nans.csv'


df = pd.read_csv(file_path)
df
## I. PREPARATION DES DONNEE DATA CLEANING
#voir les valeurs manquantes
df.info()

df.isnull().sum().sort_values(ascending=False)
df.describe(include=['O'])

#renseigner les valeurs manquantes
cat_data=[]
num_data=[]
for i,c in enumerate(df.dtypes):
    if c==object:
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()
#num_data

#pour les var categorique on va remplacer les valeurs manquantes par les valeurs qui se repetent
cat_data=cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
cat_data.isnull().sum().any()

#pour les var numerique on va remplacer les valeurs manquantes par les valeurs precedente de la meme column
num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any()



# Remplir les valeurs manquantes dans 'AgeCategory' avec une valeur par défaut avant l'extraction
df['AgeCategory'] = df['AgeCategory'].fillna('Age unknown')

# Extraire la première valeur numérique de la colonne 'AgeCategory'
cat_data['Age'] = df['AgeCategory'].str.extract('(\d+)').astype(float)

# Remplir les valeurs manquantes résultantes de l'extraction avec une valeur par défaut, comme la moyenne des âges
cat_data['Age'] = cat_data['Age'].fillna(cat_data['Age'].mean()).astype(int)

# Définir les valeurs de remplacement pour toutes les colonnes
target_value = {'Yes': 1, 'No': 0}

# Liste des colonnes à mettre à jour
columns_to_update = ['CovidPos', 'AlcoholDrinkers', 'PhysicalActivities', 'HadHeartAttack']

# Appliquer le remplacement de valeurs à chaque colonne
for column in columns_to_update:
    cat_data[column] = cat_data[column].map(target_value)

# Créer la variable target pour la colonne spécifique 'HadHeartAttack'
target_column = 'HadHeartAttack'
target = cat_data[target_column]



# Remplacer les valeurs catégoriques par des valeurs numérique 0,1,2...
le=LabelEncoder()
for i in cat_data:
    if i != 'Age':
        cat_data[i]=le.fit_transform(cat_data[i])
cat_data


# Concatener cat_Data et num_data et spécifier la colonne target
X=pd. concat ( [cat_data,num_data],axis=1)
y=target

#y


# II. ANALYSE DE NOTRE BDD

# Commencer par la variable target pour voir combien des personnes ayant eu une crise cardiaque
# target = cat_data['HadHeartAttack']
target_counts = target.value_counts()

# Dessiner le diagramme des crédits refusés et acceptés
plt.figure(figsize=(8,6))
sns.countplot(x='HadHeartAttack', data=cat_data)
plt.title("Diagramme des personnes ayant eu une crise cardiaque")
plt.xlabel("Had Heart Attack")
plt.ylabel("Nombre d'occurrences")
# Ajouter les annotations
yes_count = target_counts[1]
no_count = target_counts[0]
plt.text(0, no_count, f"{no_count} personnes n'ont pas eu de crise", ha='center', va='bottom')
plt.text(1, yes_count, f"{yes_count} personnes ont eu une crise", ha='center', va='bottom')

# Calculer le pourcentage des personnes ayant eu une crise cardiaque
yes = target_counts[1] / len(target)
no = target_counts[0] / len(target)
print(f'Le pourcentage des personnes ayant eu une crise cardiaque est: {yes*100}%') 
print(f'Le pourcentage des personnes n\'ayant pas eu de crise cardiaque est: {no*100}%')
plt.show()


# la base de données utilisée pour Lanalyse
df=pd.concat([cat_data, num_data],axis=1)

# Diagramme Comparaison Pour l'historique de creditimport seaborn as sns

grid = sns.FacetGrid(df, col='HadHeartAttack', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'Sex')
plt.suptitle("Possibilite d'avoir une crise cardiaque en fonction de Sex", y=1.05, fontsize=16)


grid = sns.FacetGrid(df, col='HadHeartAttack', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'GeneralHealth')
plt.suptitle("Possibilite d'avoir une crise cardiaque en fonction de l'état de santé général", y=1.05, fontsize=16)


grid = sns.FacetGrid(df, col='HadHeartAttack', height=3.2, aspect=1.6)
grid.map(sns.countplot, 'SmokerStatus')
plt.suptitle("Possibilite d'avoir une crise cardiaque en fonction de l'état de Fumer", y=1.05, fontsize=16)


## III. REALISATION DU MODELE EN CE BASANT SUR LES ALGO DE MACHINE LEARNING
# Diviser la base de données en une base de données test et d'entrainement
df.groupby('HadHeartAttack').median()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

print('X_train taille:', X_train.shape)
print('X_test taille:', X_test.shape)
print('y_train taille:', y_train.shape)
print('y_test taille:', y_test.shape)

# On va appliquer tois algorithmes Logisitic Regression, KNN, DecisionTree
models = {  
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier()
}

# La fonction de précision
def accu(y_true, y_pred, retu=False):
    acc = accuracy_score(y_true, y_pred)
    if retu:
        return acc
    else:
        print(f'La précision du modèle est : {acc}')

# Fonction d'application, tester et évaluer les modèles
def train_test_eval(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        print(name, ':')
        model.fit(X_train, y_train)
        accu(y_test, model.predict(X_test))
        print('-' * 30)

# Utilisation de la fonction avec vos données
train_test_eval(models, X_train, y_train, X_test, y_test)


# 2eme BDD à partir de la première mais avec 8 colonnes
X_2 = X[['Sex', 'Age', 'SleepHours', 'DifficultyErrands', 'SmokerStatus', 'PhysicalActivities', 'AlcoholDrinkers', 'CovidPos']]
#j'ai garder ce model X_2 parceq'on a trouver 99% dans La précision du modèle
# Diviser la base de données en une base de données test et d'entraînement
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X_2, y):
    X_train, X_test = X_2.iloc[train], X_2.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)

# Utiliser la fonction pour entraîner, tester et évaluer les modèles
train_test_eval(models, X_train, y_train, X_test, y_test)


#IV. Déploiement du modèle
# Appliquer la regression logisitique sur notre base de donnée
Classifier=LogisticRegression()
Classifier.fit(X_2,y)
# Enregistrer le modèle
pickle.dump(Classifier,open('model.pkl', 'wb'))
