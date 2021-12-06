# Pour le faire tourner: avoir les sources data dans le même dossier. Ouvrir une commande et taper:
# streamlit run Streamlit.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
# from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
# import scipy.stats
import datetime
# %matplotlib inline
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from PIL import Image
import requests
from io import BytesIO

@st.cache(suppress_st_warning=True)
def preprocessing():
    # chargement local
    df = pd.read_csv('eco2mix-regional-cons-def.csv', sep=';')

    # drop des colonnes vides ou quasiment vides (les TCO sont les taux de couverture par filière,
    # les flux sont incomplets, le TCH est inconnu)
    filter_col2 = [col for col in df if col.startswith('TCH')]
    filter_col3 = [col for col in df if col.startswith('Flux')]
    filter_col = filter_col2 + filter_col3
    df = df.drop(filter_col, axis=1)

    # drop des colonnes inutiles (techniques)
    df = df.drop(['Nature'], axis=1)

    # drop des lignes sans data de consommation ou production
    float_list = list(df.select_dtypes('float64').columns)
    df = df.dropna(subset=float_list, how='all', axis=0)
    # remplissage des NA par des 0 sur les colonnes de consommation et production
    for column in float_list:
        df[column].fillna(0, inplace=True)
    # remplissage des NAN des colonnes de taux de couverture de la consommation par le calcul correspondant
    df['TCO Thermique (%)'].fillna((df['Thermique (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    df['TCO Nucléaire (%)'].fillna((df['Nucléaire (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    df['TCO Eolien (%)'].fillna((df['Eolien (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    df['TCO Solaire (%)'].fillna((df['Solaire (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    df['TCO Hydraulique (%)'].fillna((df['Hydraulique (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    df['TCO Bioénergies (%)'].fillna((df['Bioénergies (MW)'] / df['Consommation (MW)'] * 100), inplace=True)
    # feature engineering (création de colonnes agrégées)
    df['Renouvelables (MW)'] = df['Eolien (MW)'] + df['Solaire (MW)'] + df['Hydraulique (MW)'] + \
                               df['Bioénergies (MW)'] + df['Pompage (MW)']
    df['Solde brut (MW)'] = df['Eolien (MW)'] + df['Solaire (MW)'] + df['Hydraulique (MW)'] + df['Bioénergies (MW)'] + \
                            df['Thermique (MW)'] + df['Nucléaire (MW)'] + df['Pompage (MW)'] - df['Consommation (MW)']
    df['Solde avec transferts (MW)'] = df['Eolien (MW)'] + df['Solaire (MW)'] + df['Hydraulique (MW)'] + \
                                       df['Bioénergies (MW)'] + df['Thermique (MW)'] + df['Nucléaire (MW)'] + \
                                       df['Pompage (MW)'] + df['Ech. physiques (MW)'] - df['Consommation (MW)']
    # remarque sur le pompage: uniquement des valeurs négatives donc représente la
    # consommation d'électricité pour alimenter les pompes des stations de transfert d'énergie (STEP). La production
    # doit alors être classifiée dans l'hydraulique.
    # Taux de couvertures de la consommation
    df['Taux Couverture (MW)'] = (df['Eolien (MW)'] + df['Solaire (MW)'] + df['Hydraulique (MW)'] +
                                  df['Bioénergies (MW)'] + df['Thermique (MW)'] + df['Nucléaire (MW)'] +
                                  df['Pompage (MW)']) / df['Consommation (MW)']
    df['Taux Couverture Renouvelables (MW)'] = df['Renouvelables (MW)'] / df['Consommation (MW)']

    # Mise au format date et extraction des éléments de la date en tant que colonnes
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Jour'] = df['Date'].dt.day
    df['Mois'] = df['Date'].dt.month
    df['Trimestre'] = pd.PeriodIndex(df['Date'], freq='Q').astype('string') # alternative: ((x.month-1)//3) +1
    df['Année'] = df['Date'].dt.year
    df['Heure'] = pd.to_datetime(df['Heure'], format='%H:%M').dt.time #alternative: format='%H:%M:%S'
    # df['Date - Heure'] = pd.to_datetime(df['Date - Heure'], utc=True)
    return df


def series_temp():
    # Groupby Month - modèle utilisé ici
    temp_df = df.drop(['Code INSEE région'],axis=1)
    temp_df = temp_df.groupby(['Date']).sum().reset_index()
    temp_df.set_index('Date', inplace=True)
    temp_df = temp_df.groupby(pd.Grouper(freq='M')).sum()

    if selected_type == 'Renouvelables (MW)':
        time_serie = temp_df['Renouvelables (MW)']
        time_serie_log = np.log(time_serie)
        model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        slider = st.sidebar.slider("Mois en forecast", min_value=0, max_value=60, value=28, step=1)

    elif selected_type == 'Nucléaire (MW)':
        time_serie = temp_df['Nucléaire (MW)']
        time_serie_log = np.log(time_serie)
        model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(2, 1, 0, 12))
        slider = st.sidebar.slider("Mois en forecast", min_value=0, max_value=60, value=28, step=1)

    elif selected_type == 'Thermique (MW)':
        time_serie = temp_df['Thermique (MW)']
        time_serie_log = np.log(time_serie)
        model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        slider = st.sidebar.slider("Mois en forecast", min_value=0, max_value=60, value=28, step=1)

    elif selected_type == 'Consommation (MW)':
        time_serie = temp_df['Consommation (MW)']
        time_serie_short = time_serie.loc['2013-1-1':'2019-12-31']
        time_serie_log = np.log(time_serie_short)
        model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        slider = st.sidebar.slider("Mois en forecast", min_value=0, max_value=60, value=28, step=1)

    season_effect = seasonal_decompose(time_serie, model='multiplicative')
    figure = season_effect.plot()
    st.write('**Seasonal Decompose, Logarithmique**')
    st.write("Vérifiez les évolutions passées avec les décompositions. Points d'attention:")
    """
    - Ruptures des tendances long terme en 2020 avec la pandémie COVID-19 avec une chute de la consommation et de certaines productions liées, sauf pour les renouvelables en hausse accélérée
    - Stationnarité des résidus globalement assurée
    """
    st.pyplot(figure)

    model_fitted = model.fit()
    # model_fitted.summary()
    pred = np.exp(model_fitted.predict(102, 102+slider))
    time_serie_pred = pd.concat([time_serie, pred])
    st.write('**Forecast**')
    figure = plt.figure(figsize=(15, 5))
    plt.plot(time_serie_pred)
    plt.axvline(x=datetime.date(2021, 6, 30), color='red')
    st.write("Modifiez l'horizon de prévision à partir de juin 2021 (trait rouge) sur le slider à gauche")
    st.pyplot(figure)


def context_series_temp():
    st.subheader("Quelle va être l'évolution du mix énergétique français?")
    st.write("Les données fournies étant chronologiques de 2013 à 2021, il est possible d'effectuer des analyses en "
             "series temporelles grâce aux modèles SARIMAX. Vous pourrez ici manipuler les résultats des 4 modèles "
             "présents, les objectifs pouvant être :")
    """
    - l'étude de la **saisonnalité** des données grâce à une décomposition statistique
    - la **prévision** des données futures sur la base des tendances passées
    """
    st.markdown("---")
    st.write("Une des problématiques du projet est de mesurer **l'évolution du mix de production français** pour avoir"
             " une idée de l'impact des renouvelables et leur futur possible en France.")
    st.write("Vous trouverez le résultat de cette analyse dans l'onglet **'Tendances'**, à comparer avec la situation "
             "actuelle ci-dessous où on constate la montée en puissance progressive des renouvelables ainsi que la "
             "prédominance du nucléaire dans le mix.")
    # response1 = requests.get('https://github.com/DataScientest-Studio/Pynergy/raw/main/Images/Explo%20-%203%20Couverture%20Renouvelables.png')
    # response2 = requests.get('https://raw.githubusercontent.com/DataScientest-Studio/Pynergy/main/Images/Explo%206%20-%20Mix_energetique_par_annee.PNG?raw=true')
    image1 = Image.open('Explo - 3 Couverture Renouvelables.png')
    image2 = Image.open('Explo 6 - Mix_energetique_par_annee.PNG')
    st.image(image1)
    st.image(image2)


def tendances():
    # Groupby Month - modèle utilisé ici
    temp_df = df.drop(['Code INSEE région'],axis=1)
    temp_df = temp_df.groupby(['Date']).sum().reset_index()
    temp_df.set_index('Date', inplace=True)
    temp_df = temp_df.groupby(pd.Grouper(freq='M')).sum()
    slider = st.sidebar.slider("Mois en forecast", min_value=0, max_value=60, value=28, step=1)

    # Calcul des series
    time_serie = temp_df['Renouvelables (MW)']
    time_serie_log = np.log(time_serie)
    model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    model_fitted = model.fit()
    pred = np.exp(model_fitted.predict(102, 102+slider))
    frame = {'Renouvelables (MW)': pred}
    pred_df = pd.DataFrame(frame, index=pred.index)

    time_serie = temp_df['Nucléaire (MW)']
    time_serie_log = np.log(time_serie)
    model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(2, 1, 0, 12))
    model_fitted = model.fit()
    pred = np.exp(model_fitted.predict(102, 102 + slider))
    pred_df['Nucléaire (MW)'] = pred

    time_serie = temp_df['Thermique (MW)']
    time_serie_log = np.log(time_serie)
    model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fitted = model.fit()
    pred = np.exp(model_fitted.predict(102, 102 + slider))
    pred_df['Thermique (MW)'] = pred

    time_serie = temp_df['Consommation (MW)']
    time_serie_short = time_serie.loc['2013-1-1':'2019-12-31']
    time_serie_log = np.log(time_serie_short)
    model = sm.tsa.SARIMAX(time_serie_log, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    model_fitted = model.fit()
    pred = np.exp(model_fitted.predict(102, 102 + slider))
    pred_df['Consommation (MW)'] = pred

    # 1er graph
    union = temp_df[['Renouvelables (MW)', 'Consommation (MW)']]
    couv_df = pd.concat([union, pred_df], axis=0)
    couv_df['Autre_conso'] = couv_df['Consommation (MW)'] - couv_df['Renouvelables (MW)']
    couv_df['Taux Couverture Renouvelables (MW)'] = couv_df['Renouvelables (MW)'] / couv_df['Consommation (MW)']
    graph_df = couv_df.loc['2019-1-31':]
    # barwidth = 25
    fig = plt.figure(figsize=(22, 10))
    ax2 = fig.add_subplot(111)
    ax2.plot(graph_df['Taux Couverture Renouvelables (MW)'], marker='D', color="limegreen", linewidth=4)
    ax2.axvline(x=datetime.date(2021, 7, 15), color='red', linewidth=4)
    ax2.set_title('Taux de couverture de la conso. par les renouvelables', fontsize=26)
    ax2.tick_params(labelsize=16)
    # ax2.set_xticks(graph_df.index)
    # ax2.set_xticklabels(graph_df.index.to_period('M'))
    # ax2.tick_params('x', labelrotation = 90)
    ax2.set_yticks(np.arange(0, 0.5, 0.05))
    ax2.set_yticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%'], fontsize=20)
    ax2.set_ylabel('Pourcentage', fontsize=26)
    ax2.tick_params('y', labelsize=20)
    ax2.grid()
    st.write('**Prévision**: hausse régulière de la couverture de la consommation par les renouvelables mais '
             'conditionnel au maintien des tendances passées (effet pandémie?)')
    st.pyplot(fig)

    # 2ème graph
    union = temp_df[['Renouvelables (MW)', 'Consommation (MW)', 'Nucléaire (MW)', 'Thermique (MW)']]
    mix_df = pd.concat([union, pred_df], axis=0)
    mix_df = mix_df.reset_index()
    mix_df['year'] = mix_df['index'].dt.year
    mix_df = mix_df.groupby(['year']).sum().reset_index()
    mix_df['Production'] = mix_df['Thermique (MW)'] + mix_df['Nucléaire (MW)'] + mix_df['Renouvelables (MW)']
    mix_df['Thermique'] = mix_df['Thermique (MW)'] / mix_df['Production']
    mix_df['Nucléaire'] = mix_df['Nucléaire (MW)'] / mix_df['Production']
    mix_df['Renouvelables'] = mix_df['Renouvelables (MW)'] / mix_df['Production']
    barwidth = 0.5
    fig2 = plt.figure(figsize=(22, 16))
    ax4 = fig2.add_subplot(111)
    graph1 = ax4.bar(x=mix_df['year'], height=mix_df['Nucléaire'], bottom=mix_df['Thermique'] + mix_df['Renouvelables'],
                     label='Nucléaire', width=barwidth, color='lightskyblue')
    graph2 = ax4.bar(x=mix_df['year'], height=mix_df['Thermique'], bottom=mix_df['Renouvelables'],
                     label='Thermique', width=barwidth, color='tomato')
    graph3 = ax4.bar(x=mix_df['year'], height=mix_df['Renouvelables'],
                     label='Renouvelables', width=barwidth, color='greenyellow')

    ax4.set_facecolor('whitesmoke')
    ax4.set_title('Mix énergétique par année', fontsize=26)
    ax4.tick_params(labelsize=20)
    ax4.set_xticks(mix_df['year'])
    ax4.set_xticklabels(mix_df['year'])
    # ax4.tick_params('x', labelrotation=90)
    ax4.set_ylim([0, 1])
    ax4.set_yticks(np.arange(0, 1.1, 0.1))
    ax4.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax4.set_ylabel('% de production', fontsize=26)

    # mise en place des pourcentages dans le graphique
    graph_list = [graph1, graph2, graph3]
    label_list = ['Nucléaire', 'Thermique', 'Renouvelables']
    for count, item in enumerate(graph_list):
        i = 0
        for p in item:
            idx = mix_df.index[i]
            label = label_list[count]
            percentage = "{:.0f}".format(round(mix_df.loc[idx, label] * 100, 0))
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height() / 2
            ax4.text(x=x, y=y, s=f'{percentage}%', ha='center', color='black', weight='bold', size=20)
            i += 1
    ax4.legend(bbox_to_anchor=(0.8, 1), fontsize=20)
    st.write("**Prévision**: hausse régulière des renouvelables, le nucléaire étant la variable d'ajustement. Il reste "
             "cependant dominant")
    st.pyplot(fig2)


def context_classification():
    st.subheader("Peut-on identifier les risques de blackout au niveau régional?")
    st.write("Sur la base des données météorologiques et du réseau électrique, il est possible d'identifier "
             "les **jours en tension** sur le réseau. Ce modèle vous permet de jouer avec les variables "
             "météorologiques et temporelles pour prévoir le solde électrique brut d'une région, classifié "
             "ainsi dans l'ordre croissant:")
    """
    - ***Déficit 3*** = -600,000 MW à -400,000 MW
    - ***Déficit 2*** = -400,000 MW à -200,000 MW
    - ***Déficit 1*** = -200,000 MW à 0 MW
    - ***Excédent 1*** = 0 MW à +200,000 MW
    - ***Excédent 2*** = +200,000 MW à +400,000 MW
    - ***Excédent 3*** = +400,000 MW à +600,000 MW
    """
    st.markdown("---")
    st.write("L'onglet **'Modèle'** vous donne les résultats du modèle et la prévision "
             "effectuée à partir des sliders situés sur la gauche de la page. Il s'agit d'un modèle XGBoost ayant "
             "donné les meilleurs résultats lors de l'entraînement")


def classification_extremes():
    #import des données météo
    meteo_df = pd.read_csv('rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv', sep=';')
    meteo_df['Date - Heure'] = pd.to_datetime(meteo_df['Date'], utc=True)
    meteo_df['Date'] = meteo_df['Date - Heure'].dt.date
    meteo_df['Date'] = pd.to_datetime(meteo_df['Date'], utc=True)
    meteo_df.sort_values(by=['Date'], inplace=True)
    meteo_df = meteo_df.groupby(['Date', 'Code INSEE région', 'Région']).mean().reset_index()

    sun_df = pd.read_csv('temperature-quotidienne-regionale.csv', sep=';')
    sun_df['Date'] = pd.to_datetime(sun_df['date'], utc=True)
    sun_df.drop(['date', 'region'], axis=1, inplace=True)
    sun_df.sort_values(by=['Date'], inplace=True)
    sun_df.reset_index(drop=True, inplace=True)
    sun_df.rename(columns={"code_insee_region": "Code INSEE région"}, inplace=True)

    temp_df = df.reset_index(drop=True)
    temp_df = temp_df.groupby(['Date', 'Code INSEE région']).sum().reset_index()
    temp_df['Date'] = pd.to_datetime(temp_df['Date'], utc=True)
    temp_df.drop(['TCO Thermique (%)', 'TCO Nucléaire (%)', 'TCO Eolien (%)', 'TCO Solaire (%)', 'TCO Hydraulique (%)',
                  'TCO Bioénergies (%)', 'Taux Couverture (MW)', 'Taux Couverture Renouvelables (MW)', 'Weekday',
                  'Jour', 'Mois', 'Année'],
                 axis=1, inplace=True)

    temp_df = temp_df.merge(right=meteo_df, on=['Date', 'Code INSEE région'], how='inner')
    temp_df = temp_df.merge(right=sun_df, on=['Date', 'Code INSEE région'], how='inner')
    temp_df['Mois'] = temp_df['Date'].dt.month
    region = pd.get_dummies(temp_df['Région'], prefix='Region').set_index(temp_df['Date'])
    mois = pd.get_dummies(temp_df['Mois'], prefix='Mois').set_index(temp_df['Date'])
    temp_df.set_index(['Date'], inplace=True)
    temp_df.drop(['Mois', 'Région', 'Code INSEE région'], axis=1, inplace=True)
    temp_df = pd.concat([temp_df, region], axis=1)  # , drop_first=True
    temp_df = pd.concat([temp_df, mois], axis=1)  # , drop_first=True
    temp_df.drop(['tmin', 'tmax', 'Consommation (MW)', 'Thermique (MW)', 'Nucléaire (MW)', 'Hydraulique (MW)',
                  'Eolien (MW)', 'Solaire (MW)', 'Pompage (MW)', 'Bioénergies (MW)', 'Ech. physiques (MW)',
                  'Renouvelables (MW)', 'Solde avec transferts (MW)'], axis=1, inplace=True)

    solde_brut = pd.cut(x=temp_df['Solde brut (MW)'], bins=[-600000, -400000, -200000, 0, 200000, 400000, 600000],
                       labels = ['Déficit3', 'Déficit2', 'Déficit1', 'Excédent1', 'Excédent2', 'Excédent3'])
    sb = pd.Series(solde_brut, name='Solde Brut')
    temp_df = pd.concat([temp_df, sb], axis=1)
    temp_df.drop('Solde brut (MW)', axis=1, inplace=True)

    # Modèle
    data = temp_df.drop(['Solde Brut', 'Region_Bretagne', 'Mois_3'], axis=1)
    target = temp_df['Solde Brut']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    rOs = RandomOverSampler()
    X_ro, y_ro = rOs.fit_resample(X_train, y_train)
    y_ro_xgb = y_ro.replace(['Déficit3', 'Déficit2', 'Déficit1', 'Excédent1', 'Excédent2', 'Excédent3'],
                            [0, 1, 2, 3, 4, 5])

    clf_ros_xgb = xgb.XGBClassifier(objective='multi:softprob', num_class=6, use_label_encoder=False,
                                    eval_metric='mlogloss', gamma = 0.1, learning_rate = 0.2, max_depth = 10,
                                    n_estimators = 50)
    clf_ros_xgb.fit(X_ro, y_ro_xgb)

    y_pred = clf_ros_xgb.predict(X_test)
    y_pred = np.where(y_pred == 0, 'Déficit3', y_pred)
    y_pred = np.where(y_pred == '1', 'Déficit2', y_pred)
    y_pred = np.where(y_pred == '2', 'Déficit1', y_pred)
    y_pred = np.where(y_pred == '3', 'Excédent1', y_pred)
    y_pred = np.where(y_pred == '4', 'Excédent2', y_pred)
    y_pred = np.where(y_pred == '5', 'Excédent3', y_pred)
    crosstab = pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédit'])
    st.write('Ce modèle final donne des rappels > 75% pour toute catégorie et une accuracy à 83%. Les erreurs de '
             'classification se font pour la plupart avec la classe adjacente.')
    st.write("Paramètres: {'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 50}")
    st.write('**Matrice de confusion**')
    st.table(crosstab)
    scoring = classification_report(y_test, y_pred)
    st.write('**Classification Report**')
    st.text(scoring)
    # return data, scaler, clf_ros_xgb

    # Nouvelle prédiction
    new_data = data.iloc[0, :]
    colnames = data.columns
    new_data['Vitesse du vent à 100m (m/s)'] = slider_wind
    new_data['Rayonnement solaire global (W/m2)'] = slider_sun
    new_data['tmoy'] = slider_temp
    new_data.iloc[3:] = 0
    for i in range(3,14):
        if selected_region == 'Bretagne':
            break
        if colnames[i] == 'Region_'+selected_region:
            new_data.iloc[i] = 1
    for i in range(14,25):
        if slider_mois == 3:
            break
        if colnames[i] == 'Mois_'+str(slider_mois):
            new_data.iloc[i] = 1
    new_data = new_data.values.reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)
    new_pred = clf_ros_xgb.predict(new_data_scaled)
    new_pred = np.where(new_pred == 0, 'Déficit3 = -600000/-400000', new_pred)
    new_pred = np.where(new_pred == '1', 'Déficit2 = -400000/-200000', new_pred)
    new_pred = np.where(new_pred == '2', 'Déficit1 = -200000/0', new_pred)
    new_pred = np.where(new_pred == '3', 'Excédent1 = 0/200000', new_pred)
    new_pred = np.where(new_pred == '4', 'Excédent2 = 200000/400000', new_pred)
    new_pred = np.where(new_pred == '5', 'Excédent3 = 400000/600000', new_pred)
    st.write('**Nouvelle prédiction**')
    st.write("*Classification du modèle en fonction des sliders sur la gauche. Dans l'ordre d'importance: région, "
             "météo, mois*")
    st.write('Temp = ' + str(slider_temp) + ' ; Vent = ' + str(slider_wind) + ' ; Soleil = ' + str(slider_sun))
    st.write('Mois = ' + str(slider_mois) + ' ; Région = ' + selected_region)
    new_text = f'***{new_pred[0]}***'
    st.write(new_text)


# Code programme
df = preprocessing()
st.title("Projet Energie")
model_list = ['Series temporelles', 'Classification'] #, 'Clustering', 'Régressions'
selected_model = st.sidebar.selectbox("Modèle", model_list)

if selected_model == 'Series temporelles':
    st.subheader("Séries Temporelles (Echelle nationale)")
    type_list = ['Contexte', 'Renouvelables (MW)', 'Nucléaire (MW)', 'Thermique (MW)', 'Consommation (MW)', 'Tendances']
    selected_type = st.sidebar.selectbox("Type", type_list)
    if selected_type == 'Tendances':
        tendances()
    elif selected_type == 'Contexte':
        context_series_temp()
    else:
        series_temp()

elif selected_model == 'Classification':
    st.subheader("Classification des soldes quotidiens")
    page_list = ['Contexte', 'Modèle']
    selected_page = st.sidebar.selectbox("Onglet", page_list)
    slider_temp = st.sidebar.slider("Température moyenne", min_value=-10.0, max_value=45.0, value=18.0, step=0.1)
    slider_wind = st.sidebar.slider("Vitesse du vent en m/s", min_value=1.0, max_value=18.0, value=1.0, step=0.1)
    slider_sun = st.sidebar.slider("Rayonnement solaire (W/m²)", min_value=0, max_value=900, value=0, step=1)
    slider_mois = st.sidebar.slider("Mois", min_value=1, max_value=12, value=1, step=1)
    region_list = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de loire', 'Grand Est',
                   'Hauts-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
                   "Provence-Alpes-Côte d'Azur", 'Île-de-France']
    selected_region = st.sidebar.selectbox("Région", region_list)
    # data, scaler, clf_ros_xgb = classification_extremes()
    if selected_page == 'Contexte':
        context_classification()
    else:
        classification_extremes()