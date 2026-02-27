# Import des bibliothèques

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt #on importe pas tout matplotlib car trop gros et non utile 
from sklearn.model_selection import train_test_split #sers a spliter la base de données en 3, train/test/validation

# Import de la base de données et nettoyage

df = pd.read_csv(r"C:\Users\SCD UM\Desktop\student.csv", sep=";") 
display(df.head()) #on regarde si l'on a bien importé
print(df.isna().sum()) #on va compter le nombre de N/A il n'est pas censé en avoir mais on regarde quand même
print(df.dtypes) #on regarde si nos données sont du bon type
df = df.copy() #on copie la base de données pour éviter de modifier le fichier existant

df.drop(columns=["student_id"],inplace=True) #nous supprimons la colone student_id qui est inutile et qui va nous faire régresser sur un ID, ce qui est inutile

# Nous avons choisis de winsorizer nos valeurs extrèmes, nous allons donc supprimer nos 1% de nos données les plus basses et hautes pour éviter les valeurs extrèmes
def winsorize(s, q_low=0.01, q_high=0.99):
    return s.clip(s.quantile(q_low), s.quantile(q_high))
    
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = winsorize(df[col])

# Analyse de la corrélation entre les variables numériques
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10,8))
plt.imshow(corr,cmap="coolwarm") #permet d'afficher les couleurs en fonction de l'intensité de la corrélation
plt.colorbar() #légende
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matrice des corrélations")
plt.tight_layout() #évite que le titre et que la légende soit coupée
plt.show()

# On va spliter nos données en train/validation/test

x = df.drop(columns=["burnout_level"]) #on enlève la variable expliquée des données d'entrainement
y = df["burnout_level"].values #on crée notre dataframe y avec les données expliquées

x_train_full, x_test, y_train_full, y_test = train_test_split(x,y,test_size=0.05, random_state=42)
# ce code va séparer nos dataframes en 2, un jeu de données pour l'entrainement et un pour le test
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size = 0.15, random_state=42)
# ce code va séparer nos dataframes en 2, un jeu de données pour le vrais entrainement et un pour la validation

#Enregistrement de dos données dans des tableaux excel sous forme csv pour s'en servir plus tard
x_train.to_csv("x_train.csv", index=False)
x_val.to_csv("x_val.csv", index=False)
x_test.to_csv("x_test.csv", index=False)

pd.Series(y_train).to_csv("y_train.csv", index=False)
pd.Series(y_val).to_csv("y_val.csv", index=False)
pd.Series(y_test).to_csv("y_test.csv", index=False)
