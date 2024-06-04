# importer les librairies
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt

# chargez les modeles
linear_model = joblib.load('Modeles/linear_model.pkl')
kmeans = joblib.load('Modeles/kmeans_model.pkl')
logistic_model = joblib.load('Modeles/logistic_model.pkl')
scaler = joblib.load('Modeles/scaler.pkl')

# remplacer les nan par linear regression
def handle_missing_values(data, linear_model):
    df_na = data[data["margin_low"].isnull()]
    if not df_na.empty:
        X_na = df_na[["margin_up", "is_genuine"]]
        predicted_na = linear_model.predict(X_na)
        data.loc[data["margin_low"].isnull(), "margin_low"] = predicted_na
    return data

# Fonction pour predire le billet
def fake_detect(data, linear_model, kmeans, logistic_model, scaler):
    data = handle_missing_values(data, linear_model)
    features = data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
    
    # Normalisation des données
    features_scaled = scaler.transform(features)
    
    # clustering k means()
    clusters = kmeans.predict(features_scaled)
    
    # Ajout des clusters
    features['cluster'] = clusters
    
    # Prediction avec la logistic regression
    predictions = logistic_model.predict(features)
    probas = logistic_model.predict_proba(features)
    
    features['Probas_faux'] = probas[:, 0]
    features['Probas_vrais'] = probas[:, 1]
    features["Predictions"] = predictions
    features["Nature_billet"] = features["Predictions"].copy()
    features["Nature_billet"].replace([0, 1], ["Faux Billet", "Vrai Billet"], inplace = True)
    
    return features

# Application de ML
st.title('Application de Machine Learning Pour la détéction de faux billet')

# A propos de l'application
st.markdown("""
Cet outil est un algorithme de détection automatique de faux billets permettant de lire un fichier contenant les dimensions de plusieurs billets et de déterminer s'ils sont authentiques en se basant sur leurs seules dimensions.

L'algorithme utilise un modèle de clustering puis un modele de logistic regression entraîné et optimisé grâce à un fichier d'exemple contenant 1500 billets dont 1000 étaient vrais et 500 étaient faux.

Il a été développé en Python grâce à la librairie de Machine Learning scikit-learn.
""")


# Image
st.image("images/image1.jpg", caption="Billet CFA", use_column_width=True)

# chargez le fichier
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    # lire le fichier csv
    data = pd.read_csv(uploaded_file, sep=",")
    
    # affichez les données
    st.write("Données téléchargées :")
    st.write(data.head())
    
    # Realisez la prédiction
    results = fake_detect(data, linear_model, kmeans, logistic_model, scaler)
    
    # Suppression des colonnes
    results = results.drop(columns=['cluster', 'Probas_faux', 'Probas_vrais'])
    
    # Affichez les resultats
    st.write("Résultats des prédictions :")
    st.write(results)
    
    
    
    # Visualisez les resultats
    #st.write("Graphique des résultats :")
    #st.bar_chart(results["Nature_billet"].value_counts())
    #sns.countplot(x="Nature_billet", data=X, palette=["darkslategrey","bisque"], legend=False)
    
    
    # Visualisez les resultats
    import plotly.express as px
    st.write("Diagramme des résultats avec Plotly :")
    fig = px.histogram(results, x="Nature_billet", color="Nature_billet", barmode="group")
    st.plotly_chart(fig)
    
    # Display Fake bills and Real bills
    st.write("Faux Billet:")
    st.write(results[results["Nature_billet"] == "Faux Billet"])
    st.write("Vrai Billet:")
    st.write(results[results["Nature_billet"] == "Vrai Billet"])
    
    # Calculate and display the accuracy
    #accuracy = accuracy_score(data['is_genuine'], results['Predictions'])
    #st.write(f"Précision : {accuracy:.2f}")
    
    # Compute and display the confusion matrix
    #conf_matrix = confusion_matrix(data['is_genuine'], results['Predictions'])
    #st.write("Matrice de confusion :")
    #st.write(conf_matrix)

# Run the Streamlit app
if __name__ == '__main__':
    st.write("Prêt à détecter les faux billets !")
