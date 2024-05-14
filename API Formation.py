from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from textblob import TextBlob
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


app = Flask(__name__)
CORS(app)

# Charger les données de formation
data_formation = pd.read_excel('C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa\\Monaco-all-result.xlsx')

# Charger les données de classement
data_rank = pd.read_excel('C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa//understat.com.xls')
club = data_rank['Squad'].unique()
# Charger les données des joueurs
data_players = pd.read_excel('C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa\\joueursssss.xlsx')
joueurs_value = data_players['short_name'].unique()
# Charger les données des Commentaire
data = pd.read_excel('C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa\\data.xlsx')

# Charger le modèle SVM depuis le fichier pickle
model = joblib.load('C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa\\model_Bless.pkl')

# Charger les données nécessaires
data_bless = pd.read_excel("C:\\Users\\Nadhir\\Desktop\\2ème Annéé Cycle Ingénieur\\2EME SEMESTRE\\PROJET_BI\\AngularML\\dataaaa\\DimBlessure.xlsx")
data_bless.dropna(subset=['Debut', 'Fin', 'Short Name', 'Température'], inplace=True)
data_bless['Debut'] = pd.to_datetime(data_bless['Debut'], dayfirst=True)
data_bless['Fin'] = pd.to_datetime(data_bless['Fin'], dayfirst=True, errors='coerce')
data_bless['Duree'] = (data_bless['Fin'] - data_bless['Debut']).dt.days
blessures_total_par_joueur = data_bless.groupby('Short Name').size()
blessures_par_joueur_par_saison = data_bless.groupby(['Short Name', 'Saison']).size()
features = pd.DataFrame({
    'Blessures_Total': blessures_total_par_joueur,
    'Blessures_Max_Saison': blessures_par_joueur_par_saison.groupby('Short Name').max(),
    'Blessures_Moyenne_Saison': blessures_par_joueur_par_saison.groupby('Short Name').mean(),
    'Température': data_bless.groupby('Short Name')['Température'].mean()
}).reset_index()
features['Cible'] = features.apply(lambda x: 'Blessé' if blessures_total_par_joueur[x['Short Name']] > 5 or blessures_par_joueur_par_saison[x['Short Name']].max() > 2 else 'Non blessé', axis=1)

# Liste des noms de joueurs disponibles
joueurs_disponibles = data_bless['Short Name'].unique()

# Prétraitement des données de formation
data_formation['FormationHome'].replace('-', '', inplace=True)
data_formation.dropna(inplace=True)
data_formation['HomeTeamScore'] = pd.to_numeric(data_formation['HomeTeamScore'], errors='coerce')
data_formation['AwayTeamScore'] = pd.to_numeric(data_formation['AwayTeamScore'], errors='coerce')

X_formation = data_formation[['HomeTeamScore', 'AwayTeamScore']]
y_formation = data_formation['FormationHome']
X_train_formation, X_test_formation, y_train_formation, y_test_formation = train_test_split(X_formation, y_formation, test_size=0.2, random_state=42)

model_formation = DecisionTreeClassifier(random_state=42)
model_formation.fit(X_train_formation, y_train_formation)

# Prétraitement des données de classement
current_year_rank = data_rank['Season'].max()
previous_years_data_rank = data_rank[data_rank['Season'] < current_year_rank]

features_rank = ['Season', 'Squad', 'W', 'L', 'D', 'scored', 'missed']
target_rank = 'LgRk'

X_rank = previous_years_data_rank[features_rank]
y_rank = previous_years_data_rank[target_rank]

X_train_rank, _, y_train_rank, _ = train_test_split(X_rank, y_rank, test_size=0.2, random_state=42)

column_transformer_rank = ColumnTransformer(
    [('onehot', OneHotEncoder(handle_unknown='ignore'), ['Squad'])],
    remainder='passthrough'
)

X_train_transformed_rank = column_transformer_rank.fit_transform(X_train_rank)

model_rank = LinearRegression()
model_rank.fit(X_train_transformed_rank, y_train_rank)

# Prétraitement des données des joueurs
data_historique = data_players[data_players['Season'] < 2024]
data_prochaine_annee = data_players[data_players['Season'] == 2024]

X_historique = data_historique[['age', 'overall', 'passing']]
y_historique = data_historique['value_eur']

imputer = SimpleImputer(strategy='mean')
X_historique_imputed = imputer.fit_transform(X_historique)

model_players = RandomForestRegressor(random_state=42)
model_players.fit(X_historique_imputed, y_historique)

# Route pour les noms de joueurs disponibles (méthode GET)
@app.route('/players', methods=['GET'])
def get_players():
    return jsonify({"players": list(joueurs_disponibles)})

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    # sourcery skip: assign-if-exp, boolean-if-exp-identity, introduce-default-else, move-assign-in-block, remove-unnecessary-cast, remove-unnecessary-else, swap-if-else-branches
    content = request.get_json()
    nom_joueur = content['nom_joueur']
    if nom_joueur in features['Short Name'].values:
        joueur_data = features[features['Short Name'] == nom_joueur].drop(['Short Name', 'Cible'], axis=1)
        prediction = model.predict(joueur_data)
        tendance_blessure = False
        blessure_recente = False
        type_blessure_grave = False
        
        if prediction == "Blessé":
            joueur_blessures = data_bless[data_bless['Short Name'] == nom_joueur]
            if (datetime.datetime.now() - joueur_blessures['Fin'].max()).days <= 30:
                blessure_recente = True
                if any(joueur_blessures['Duree'] > 30):
                    type_blessure_grave = True
        
        if blessures_total_par_joueur[nom_joueur] > 5 or blessures_par_joueur_par_saison[nom_joueur].max() > 2:
            tendance_blessure = True
        
        if tendance_blessure and blessure_recente:
            raison = "Tendance à être souvent blessé et blessure récente"
        elif tendance_blessure:
            raison = "Tendance à être souvent blessé"
        elif blessure_recente:
            raison = "Blessure récente"
        else:
            raison = "Pas de tendance à être souvent blessé"
        pre=prediction[0]
        gravite_blessure = "Grave" if type_blessure_grave else "Pas grave"
        
        return jsonify({"Joueur": nom_joueur, "Prediction": pre, "Raison": raison, "Gravite de la blessure": gravite_blessure})
    else:
        return jsonify({"error": "Ce joueur n'est pas présent dans les données."})
    
############################################################################
@app.route('/predict_formation', methods=['GET'])
def predict_formation():
    formation_prediction = model_formation.predict([[X_formation['HomeTeamScore'].mean(), X_formation['AwayTeamScore'].mean()]])
    return jsonify({'formation_prediction': formation_prediction.tolist()})

# Route pour les noms de joueurs disponibles (méthode GET)
@app.route('/club', methods=['GET'])
def get_club():
    return jsonify({"players": list(club)})

@app.route('/predict_rank', methods=['POST'])
def predict_rank():
    data = request.get_json()
    team = data['team']
    team_data = previous_years_data_rank[previous_years_data_rank['Squad'] == team]
    if team_data.empty:
        return jsonify({'error': f"No data found for team {team}"}), 404
    
    avg_performances = team_data[['W', 'L', 'D', 'scored', 'missed']].mean().to_dict()
    current_season = previous_years_data_rank['Season'].max()
    new_team = pd.DataFrame({
        'W': [avg_performances['W']],
        'L': [avg_performances['L']],
        'D': [avg_performances['D']],
        'scored': [avg_performances['scored']],
        'missed': [avg_performances['missed']],
        'Squad': [team],
        'Season': [current_season]
    })
    new_team_transformed = column_transformer_rank.transform(new_team)
    predicted_rank = model_rank.predict(new_team_transformed)[0]
    return jsonify({'predicted_rank': round(predicted_rank)}), 200

# Route pour les noms de joueurs disponibles (méthode GET)
@app.route('/playersValue', methods=['GET'])
def playersValue():
    return jsonify({"players": list(joueurs_value)})


@app.route('/predict_player_value', methods=['POST'])
def predict_player_value():
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    req_data = request.get_json()
    if 'player_name' in req_data:
        player_name = req_data['player_name']
        player_data = data_prochaine_annee[data_prochaine_annee['short_name'] == player_name]
        if not player_data.empty:
            X_player = player_data[['age', 'overall', 'passing']]
            X_player_imputed = imputer.transform(X_player)
            y_pred = model_players.predict(X_player_imputed)
            return jsonify({
                'player_name': player_name,
                'predicted_value': int(y_pred[0])
            })
        else:
            return jsonify({'error': 'The specified player does not exist in the data.'}), 404
    else:
        return jsonify({'error': 'Please provide the player name in the JSON request body with key "player_name".'}), 400

@app.route('/analyse_sentiments', methods=['GET'])
def analyse_sentiments():
    comments_list = data['commentaires']
    results = []

    for commentaire in comments_list:
        blob = TextBlob(commentaire)
        sentiment = blob.sentiment
        polarite=sentiment.polarity
        if polarite > 0.5:
            returne= 'Content/Satisfait'
        elif polarite > 0:
            returne=  'Satisfait'
        elif polarite < 0:
            returne=  'Triste/Non satisfait'
        else:
            returne=  'Neutre'
        results.append({
            'commentaire': commentaire,
            'returne':returne ,
            'polarite':polarite
        })

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
