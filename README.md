# Import des bibliothèques

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt #on importe pas tout matplotlib car trop gros et non utile 
from sklearn.model_selection import train_test_split #sers a spliter la base de données en 3, train/test/validation

# Import de la base de données et nettoyage

df = pd.read_csv(r"https://raw.githubusercontent.com/nayelguermache29-creator/projetML/main/data/student.csv", sep=";") 
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
# Import des biblothèques
from sklearn.model_selection import KFold, RandomizedSearchCV #sers pour la cross validation et de la recherche d'hyperparamètres
from sklearn.compose import ColumnTransformer #permet de traiter différement les colones si elles sont numériques ou catégorielles
from sklearn.preprocessing import OneHotEncoder, StandardScaler # permet de standardiser les données numériques et de transformer les données texte en 0,1
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #permet d'importer nos métriques d'évaluation comme le MAE ou le MSE
from sklearn.ensemble import RandomForestRegressor #modèle de ML random forest
import statsmodels.api as sm #bibliothèque de statistiques

#Import de nos bases de données (si le code d'avant à été executé alors pas besoin d'executer cette partie du code)
x_train = pd.read_csv("x_train.csv")
x_val   = pd.read_csv("x_val.csv")
x_test  = pd.read_csv("x_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_val   = pd.read_csv("y_val.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

#On concatène nos données pour la recherche d'hyperparamètres avec cross validation
x_train_full = pd.concat([x_train, x_val], axis=0)
y_train_full = np.concatenate([y_train, y_val])

#preprocessing
cat_cols=x_train.select_dtypes(include=["object"]).columns.tolist()
num_cols=x_train.select_dtypes(include=[np.number]).columns.tolist()
#crée 2 catégories de colones, une catégorie objet et une numérique

preprocess = ColumnTransformer( #introduis preprocess, on s'en servira plus tard
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)


#Régression linéaire simple - OLS
x_train_ols = preprocess.fit_transform(x_train)
x_test_ols = preprocess.transform(x_test)

x_train_ols = x_train_ols if isinstance(x_train_ols, np.ndarray) else x_train_ols.toarray()
x_test_ols = x_test_ols if isinstance(x_test_ols, np.ndarray) else x_test_ols.toarray()

x_train_sm = sm.add_constant(x_train_ols)
x_test_sm = sm.add_constant(x_test_ols)

ols = sm.OLS(y_train, x_train_sm).fit() #fait la régression
y_pred_ols = ols.predict(x_test_sm) #prédit les données

#Données d'erreur
print(f"MAE={round(mean_absolute_error(y_test, y_pred_ols),3)}")
print(f"RMSE={round(np.sqrt(mean_squared_error(y_test, y_pred_ols)),3)}")
print(f"R²={round(r2_score(y_test, y_pred_ols),3)}")

#Affichage du graphique de la régression
plt.figure()
plt.scatter(y_test, y_pred_ols, alpha=0.5)
plt.title("Données prédites vs données réelles")
plt.xlabel("Données réelles")
plt.ylabel("Données prédites")
plt.show()


# Random forest

preprocess = ColumnTransformer( #on va standardiser les données et passer les non numériques en 0 et 1
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

rf_pipe = Pipeline([("prep",  preprocess), ("rf", RandomForestRegressor(random_state=42))])
# dans ce code, nous avons préparé nos données ainsi que définir notre random_state pour la reproductibilité
param_grid = {
    "rf__n_estimators": [200, 500, 800],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 5, 10]
}
#ici c'est l'hyperparamètre, nous allons tester différentes combinaisons d'arbres
#nous allons tester en tout 27 combinaisons avec différents nombre d'estimateurs, différentes profondeurs et un nb
#différent d'observations par feuille

cv = KFold(n_splits=5, shuffle=True, random_state=42) #on mélange nos données et on sépare nos données en 5 pour faire 5 train/validation sur a chaque fois 80%/20%

rf_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_grid,
    n_iter=10, #nous ne testerons que 10 itérations finalement, pas 27
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42
)
#concrètement, on crée ici toute la structure qui vas nous permettre de faire notre régression
# on a définis nos paramètres avant et maintenant nous allons tout concaténer pour ensuite s'entrainer

rf_search.fit(x_train_full, y_train_full) #on entraine maintena tnotre modèle

best_rf = rf_search.best_estimator_ #ici, on stoke notre meilleur modèle entrainé une nouvelle fois dans cette variable
y_pred_rf = best_rf.predict(x_test) #on teste notre meilleur modèle sur nos données test

#Données de performance
print(f"MAE={round(mean_absolute_error(y_test, y_pred_rf),3)}")
print(f"RMSE={round(np.sqrt(mean_squared_error(y_test, y_pred_rf)),3)}")
print(f"R²={round(r2_score(y_test, y_pred_rf),3)}")

#Résultat
plt.figure()
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.title("Données prédites vs données réelles - Random fotest")
plt.xlabel("Données réelles burnout")
plt.ylabel("Données prédites burnout")
plt.show()



# Deep learning

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


preprocess = ColumnTransformer( #on va standardiser les données et passer les non numériques en 0 et 1
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)
# on transforme nos données pour le deeplearning
x_train_dl = preprocess.fit_transform(x_train)
x_val_dl = preprocess.transform(x_val)
_test_dl = preprocess.transform(x_test)
x_test_dl = preprocess.transform(x_test)

class TabularDataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).view(-1,1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
# on crée une class avec nos objets d'entrée et de sortie pour que pytorch comprenne 

#on va créer un modèle de deep learning
class MLP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64), #64 neurones
            nn.ReLU(), #sers a faire de la non linéarité
            nn.Dropout(0.2), #20% des neuronnes sont désactivés aléatoirement pour éviter l'overfitting
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1) #transforme notre vecteur en une seule valeur 
        )
    def forward(self, x):
        return self.net(x)

#création de notre fonction de loss
class WeightedMSE(nn.Module):
    def __init__(self, threshold, alpha=2.0):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, y_pred, y_true): #décrit comment calculer la loss
        w = torch.where(y_true > self.threshold, self.alpha, 1.0) 
        return torch.mean(w * (y_pred - y_true) ** 2) #focntion de la loss

train_ds = TabularDataset(x_train_dl, y_train) #on transforme les données d'entrainement pour PyTorch
val_ds = TabularDataset(x_val_dl, y_val) #on transforme les données de validation pour PyTorch
test_ds = TabularDataset(x_test_dl, y_test) #on transforme les données de test pour PyTorch

#ici-bas, nous allons diviser nos données en différentes sous données (batch)
train_loader = DataLoader(train_ds, batch_size = 64, shuffle= True)# on mélange nos données pour que notre modèle n'aprenne pas un ordre précis
val_loader= DataLoader(val_ds,batch_size=64)#pas besoin de mélanger nos données de validation

#on continue a données des informations a notre modèle pour qu'il aprenne bien
model=MLP(x_train_dl.shape[1]) #on donne a notre modèle notre bon nombre de colones a analyser
criterion= WeightedMSE(np.quantile(y_train,0.75)) #on pénalise plus fortement les données quand le nombre est grand
optimizer=AdamW(model.parameters(), lr=1e-3) #on définis la taille du pas


#DEEP LEARNING

for epoch in range(40): #on va revoir les données d'entrainement 40 fois
    model.train()
    for xb, yb in train_loader: #on prend les données d'entrée et de sortie
        optimizer.zero_grad() #il cherche a optimiser
        loss = criterion(model(xb),yb) 
        #on va prendre les données du modèle et voir si elles sont conformes aux données réelles et on va calculer la fonctionn de loss dessus
        loss.backward() #le modèle cherche ici a voir quelle donnée lui a couté l'erreur
        optimizer.step() #le modèle modifie ses parapètres pour mieux prédire la prochaine fois
        
    if epoch%5==0:
        model.eval()
        with torch.no_grad(): #on utilise torch.no_grad pour pas utilise tour torch
            val_predites = torch.cat([model(xb) for xb, _ in val_loader])
            val_réelles = torch.cat([yb for _, yb in val_loader])
            mae = torch.mean(torch.abs(val_predites - val_réelles))
        print(f"Epoch {epoch} | Val MAE: {mae:.4f}") #toutes les 5 époques, on va regarder notre MAE pour voir comment il évolue
        
#ici-bas, on évalue le modèle
model.eval()
with torch.no_grad(): #on utilise torch.no_grad pour pas utiliser tour torch, notre modèle doit pas apprendre, juste donner les réponses
    y_pred_dl = model(torch.tensor(x_test_dl, dtype=torch.float)).numpy().flatten()

print("MAE :", round(mean_absolute_error(y_test, y_pred_dl),3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_dl)),3))
print("R2  :", round(r2_score(y_test, y_pred_dl),3))

# Plot: true vs predicted



plt.figure()
plt.scatter(y_test, y_pred_dl, alpha=0.5)
plt.title("Données réelles vs prédites")
plt.xlabel("Données réelles burnout")
plt.ylabel("Données prédites burnout")
plt.show()



#Comparaison entre les 3 modèles
results = pd.DataFrame({
    "Modele": ["OLS", "RandomForest","DeepLearning"],
    "MAE": [
        mean_absolute_error(y_test, y_pred_ols),
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_dl)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_ols)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test, y_pred_dl))
    ],
    "R2": [
        r2_score(y_test, y_pred_ols),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_dl)
    ]
})
results
