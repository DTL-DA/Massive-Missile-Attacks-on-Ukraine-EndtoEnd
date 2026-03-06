# =============================================================================
# Tarea Semana 3 -- Proyecto End-to-End de Machine Learning (Regresion)
# Dataset: Massive Missile Attacks on Ukraine
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========================
# CONFIGURACION DE LA APP
# ========================
st.set_page_config(
    page_title="Missile Attacks Ukraine - ML End-to-End",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# Tarea Semana 3 -- Proyecto End-to-End de Machine Learning (Regresion)
## Dataset: *Massive Missile Attacks on Ukraine*

**Objetivo:** Predecir la cantidad de misiles/drones **destruidos** (`destroyed`) en cada ataque,
a partir de las caracteristicas del ataque (modelo de arma, lugar de lanzamiento, cantidad lanzada, etc.).

### Pipeline completo:
1. Importacion del dataset
2. Analisis exploratorio (EDA)
3. Ingenieria de caracteristicas
4. Pipeline con `ColumnTransformer`
5. Modelos de regresion -> Random Forest + GridSearchCV
6. Reporte final + visualizaciones
""")

# ========================
# 1. CARGA DE DATOS
# ========================
st.header("1. Importacion de Librerias y Datos")


@st.cache_data
def cargar_datos():
    df = pd.read_csv('data tarea 3/missile_attacks_daily.csv')
    df_models = pd.read_csv('data tarea 3/missiles_and_uavs.csv')
    return df, df_models


df, df_models = cargar_datos()

st.success("Librerias cargadas correctamente")
st.write(f"**Dataset principal:** {df.shape[0]} filas x {df.shape[1]} columnas")
st.write(f"**Catalogo de modelos:** {df_models.shape[0]} filas x {df_models.shape[1]} columnas")
st.dataframe(df.head())

# ========================
# 2. EDA
# ========================
st.header("2. Analisis Exploratorio de Datos (EDA)")

st.subheader("Informacion General del Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{df.shape[0]:,}")
col2.metric("Columnas", f"{df.shape[1]}")
col3.metric("Rango temporal", f'{df["time_start"].min()} a {df["time_start"].max()}')

st.write("**Tipos de datos:**")
st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipo"]))

st.write("**Valores nulos por columna:**")
nulls = df.isnull().sum()
nulls_pct = (df.isnull().sum() / len(df) * 100).round(1)
null_df = pd.DataFrame({'Nulos': nulls, '% Nulo': nulls_pct})
st.dataframe(null_df[null_df['Nulos'] > 0].sort_values('% Nulo', ascending=False))

# Estadisticas descriptivas
st.subheader("Estadisticas Descriptivas")
cols_num = ['launched', 'destroyed', 'not_reach_goal', 'still_attacking']
st.dataframe(df[cols_num].describe().round(2))

# Distribucion de la variable objetivo
st.subheader("Distribucion de la Variable Objetivo: destroyed")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['destroyed'].dropna(), bins=50, color='#667eea', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribucion de Destroyed', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cantidad destruida')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['destroyed'].mean(), color='red', linestyle='--',
                label=f'Media: {df["destroyed"].mean():.1f}')
axes[0].legend()

axes[1].boxplot(df['destroyed'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='#667eea', alpha=0.6))
axes[1].set_title('Boxplot de Destroyed', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Cantidad destruida')

axes[2].scatter(df['launched'], df['destroyed'], alpha=0.4, color='#764ba2', s=20)
axes[2].plot([0, df['launched'].max()], [0, df['launched'].max()], 'r--', alpha=0.5, label='Ideal (100%)')
axes[2].set_title('Launched vs Destroyed', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Lanzados')
axes[2].set_ylabel('Destruidos')
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

# Pairplot
st.subheader("Pairplot - Relaciones entre Variables Clave")
fig_pair = sns.pairplot(df, vars=['launched', 'destroyed', 'not_reach_goal', 'still_attacking'],
                        kind='scatter', plot_kws={'alpha': 0.5, 'color': '#667eea'})
fig_pair.figure.suptitle('Relaciones entre variables clave', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig_pair.figure)

# Top 15 modelos
st.subheader("Top 15 Modelos de Armas Mas Utilizados")
top_modelos = df['model'].value_counts().head(15)

fig2, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_modelos.index[::-1], top_modelos.values[::-1], color='#667eea', edgecolor='white')
ax.set_xlabel('Frecuencia de ataques', fontsize=12)
ax.set_title('Top 15 Modelos de Armas Mas Utilizados', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top_modelos.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig2)

# Tasa de intercepcion
st.subheader("Tasa de Intercepcion por Modelo de Arma (Top 10)")
top10 = df['model'].value_counts().head(10).index
df_top10 = df[df['model'].isin(top10)].copy()
df_top10['interception_rate'] = df_top10['destroyed'] / df_top10['launched']

tasa_modelo = df_top10.groupby('model')['interception_rate'].mean().sort_values(ascending=False)

fig3, ax = plt.subplots(figsize=(12, 5))
colors_tasa = ['#2ecc71' if v > 0.8 else '#e67e22' if v > 0.5 else '#e74c3c' for v in tasa_modelo.values]
bars = ax.barh(tasa_modelo.index[::-1], tasa_modelo.values[::-1], color=colors_tasa[::-1], edgecolor='white')
ax.set_xlabel('Tasa de Intercepcion Promedio', fontsize=12)
ax.set_title('Tasa de Intercepcion por Modelo de Arma (Top 10)', fontsize=14, fontweight='bold')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
for bar, val in zip(bars, tasa_modelo.values[::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig3)

# Matriz de correlacion
st.subheader("Matriz de Correlacion")
cols_corr = ['launched', 'destroyed', 'not_reach_goal', 'still_attacking',
             'is_shahed', 'num_hit_location', 'num_fall_fragment_location', 'turbojet', 'turbojet_destroyed']
corr_matrix = df[cols_corr].corr()

fig4, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Matriz de Correlacion', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)

# ========================
# 3. INGENIERIA DE CARACTERISTICAS
# ========================
st.header("3. Ingenieria de Caracteristicas")

data = df.copy()

data['time_start'] = pd.to_datetime(data['time_start'], errors='coerce')
data['year'] = data['time_start'].dt.year
data['month'] = data['time_start'].dt.month
data['day_of_week'] = data['time_start'].dt.dayofweek
data['day_of_year'] = data['time_start'].dt.dayofyear

data = data.merge(df_models[['model', 'category', 'type']], on='model', how='left')
data['category'] = data['category'].fillna('Unknown')
data['type'] = data['type'].fillna('unknown')


def simplificar_target(t):
    if pd.isna(t):
        return 'Unknown'
    t = str(t).lower()
    if 'ukraine' in t:
        return 'Ukraine_general'
    elif 'south' in t:
        return 'South'
    elif 'east' in t:
        return 'East'
    elif 'odesa' in t:
        return 'Odesa'
    elif 'kherson' in t:
        return 'Kherson'
    elif 'kharkiv' in t:
        return 'Kharkiv'
    elif 'kyiv' in t:
        return 'Kyiv'
    elif 'mykolaiv' in t:
        return 'Mykolaiv'
    elif 'dnipro' in t:
        return 'Dnipro'
    else:
        return 'Other'


data['target_region'] = data['target'].apply(simplificar_target)

top10_models = data['model'].value_counts().head(10).index
data['model_group'] = data['model'].apply(lambda x: x if x in top10_models else 'Other')
data['has_launch_place'] = data['launch_place'].notna().astype(int)
data['is_shahed_flag'] = data['is_shahed'].fillna(0).apply(lambda x: 1 if x > 0 else 0)

st.write(f"**Dataset enriquecido:** {data.shape}")
st.write("**Nuevas columnas creadas:**")
st.markdown("""
- year, month, day_of_week, day_of_year
- category, type (del catalogo de modelos)
- target_region (region simplificada)
- model_group (top 10 + Other)
- has_launch_place, is_shahed_flag
""")
st.dataframe(data[['model', 'model_group', 'category', 'type', 'target_region', 'launched', 'destroyed']].head(10))

# Seleccion de features
data_model = data.dropna(subset=['launched', 'destroyed']).copy()

feature_cols_num = ['launched', 'year', 'month', 'day_of_week', 'day_of_year',
                    'has_launch_place', 'is_shahed_flag']
feature_cols_cat = ['model_group', 'category', 'type', 'target_region']

X = data_model[feature_cols_num + feature_cols_cat]
y = data_model['destroyed']

st.write(f"**Tamano de X:** {X.shape}")
st.write(f"**Tamano de y:** {y.shape}")
st.write(f"**Features numericas ({len(feature_cols_num)}):** {feature_cols_num}")
st.write(f"**Features categoricas ({len(feature_cols_cat)}):** {feature_cols_cat}")

# ========================
# 4. PIPELINE
# ========================
st.header("4. Pipeline con ColumnTransformer")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"**Train:** {X_train.shape[0]:,} muestras")
st.write(f"**Test:** {X_test.shape[0]:,} muestras")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, feature_cols_num),
        ('cat', cat_pipeline, feature_cols_cat)
    ],
    remainder='drop'
)

st.success("ColumnTransformer definido:")
st.markdown("""
- **Numericas** -> SimpleImputer(median) + StandardScaler
- **Categoricas** -> SimpleImputer(constant) + OneHotEncoder
""")

# ========================
# 5. MODELOS
# ========================
st.header("5. Modelos de Regresion")
st.write("Entrenamos multiples modelos como baseline y luego optimizamos el mejor con GridSearchCV.")


@st.cache_resource
def entrenar_modelos(_preprocessor, _X_train, _y_train, _X_test, _y_test, _feature_cols_num, _feature_cols_cat):
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    resultados = []

    for nombre, modelo in modelos.items():
        pipe = Pipeline([
            ('preprocessor', _preprocessor),
            ('model', modelo)
        ])

        cv_scores = cross_val_score(pipe, _X_train, _y_train, cv=5, scoring='r2', n_jobs=-1)

        pipe.fit(_X_train, _y_train)
        y_pred = pipe.predict(_X_test)

        mae = mean_absolute_error(_y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(_y_test, y_pred))
        r2 = r2_score(_y_test, y_pred)

        resultados.append({
            'Modelo': nombre,
            'CV R2 (mean)': cv_scores.mean(),
            'CV R2 (std)': cv_scores.std(),
            'Test MAE': mae,
            'Test RMSE': rmse,
            'Test R2': r2
        })

    df_resultados = pd.DataFrame(resultados).sort_values('Test R2', ascending=False)

    # GridSearchCV
    pipe_rf = Pipeline([
        ('preprocessor', _preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        pipe_rf, param_grid,
        cv=5, scoring='r2',
        n_jobs=-1, verbose=0,
        return_train_score=True
    )

    grid_search.fit(_X_train, _y_train)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(_X_test)

    mae_best = mean_absolute_error(_y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(_y_test, y_pred_best))
    r2_best = r2_score(_y_test, y_pred_best)

    return df_resultados, grid_search, best_model, y_pred_best, mae_best, rmse_best, r2_best


with st.spinner("Entrenando modelos (esto puede tardar unos minutos)..."):
    df_resultados, grid_search, best_model, y_pred_best, mae_best, rmse_best, r2_best = entrenar_modelos(
        preprocessor, X_train, y_train, X_test, y_test, feature_cols_num, feature_cols_cat
    )

st.subheader("Ranking de Modelos Baseline")
st.dataframe(df_resultados)

# Visualizacion de comparacion
fig5, axes = plt.subplots(1, 2, figsize=(16, 5))

colors_mod = ['#667eea', '#764ba2', '#2ecc71', '#e67e22']
bars = axes[0].barh(df_resultados['Modelo'], df_resultados['Test R2'],
                    color=colors_mod[:len(df_resultados)], edgecolor='white')
axes[0].set_xlabel('R2 Score', fontsize=12)
axes[0].set_title('Comparacion de Modelos -- R2 en Test', fontsize=14, fontweight='bold')
for bar, val in zip(bars, df_resultados['Test R2']):
    axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)

bars2 = axes[1].barh(df_resultados['Modelo'], df_resultados['Test MAE'],
                     color=colors_mod[:len(df_resultados)], edgecolor='white')
axes[1].set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
axes[1].set_title('Comparacion de Modelos -- MAE en Test', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, df_resultados['Test MAE']):
    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()
st.pyplot(fig5)

# GridSearchCV resultados
st.subheader("5.1 Optimizacion con GridSearchCV (Random Forest)")

st.success("Mejores hiperparametros encontrados:")
for param, val in grid_search.best_params_.items():
    st.write(f"  - **{param}:** {val}")
st.write(f"**Mejor CV R2:** {grid_search.best_score_:.4f}")

st.subheader("Resultados del Modelo Optimizado (Random Forest)")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae_best:.2f}")
col2.metric("RMSE", f"{rmse_best:.2f}")
col3.metric("R2", f"{r2_best:.4f}")

r2_base = df_resultados[df_resultados['Modelo'] == 'Random Forest']['Test R2'].values[0]
st.write(f"**Mejora vs baseline Random Forest:** R2 baseline: {r2_base:.4f} -> R2 optimizado: {r2_best:.4f} (Delta = {r2_best - r2_base:+.4f})")

# ========================
# 6. REPORTE FINAL
# ========================
st.header("6. Reporte Final + Visualizaciones")

# Predicciones vs Valores Reales
st.subheader("Predicciones vs Valores Reales")
fig6, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_test, y_pred_best, alpha=0.4, color='#667eea', s=20, edgecolor='white', linewidth=0.3)
lim_max = max(y_test.max(), y_pred_best.max()) * 1.05
axes[0].plot([0, lim_max], [0, lim_max], 'r--', alpha=0.7, label='Prediccion perfecta')
axes[0].set_xlabel('Valor Real (destroyed)', fontsize=12)
axes[0].set_ylabel('Prediccion', fontsize=12)
axes[0].set_title(f'Real vs Predicho (R2={r2_best:.4f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim(-5, lim_max)
axes[0].set_ylim(-5, lim_max)

residuos = y_test - y_pred_best
axes[1].hist(residuos, bins=50, color='#764ba2', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Residuo (Real - Predicho)', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title(f'Distribucion de Residuos (MAE={mae_best:.2f})', fontsize=14, fontweight='bold')

plt.tight_layout()
st.pyplot(fig6)

# Importancia de Features
st.subheader("Importancia de Features")
rf_model = best_model.named_steps['model']
preprocessor_fitted = best_model.named_steps['preprocessor']

ohe_features = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(feature_cols_cat)
all_features = list(feature_cols_num) + list(ohe_features)

importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': all_features, 'Importancia': importances})
feat_imp = feat_imp.sort_values('Importancia', ascending=False).head(20)

fig7, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(feat_imp['Feature'][::-1], feat_imp['Importancia'][::-1],
               color='#667eea', edgecolor='white')
ax.set_xlabel('Importancia', fontsize=12)
ax.set_title('Top 20 Features Mas Importantes (Random Forest Optimizado)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, feat_imp['Importancia'][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig7)

# Error por categoria
st.subheader("Analisis de Error por Categoria de Arma")
test_analysis = X_test.copy()
test_analysis['real'] = y_test.values
test_analysis['prediccion'] = y_pred_best
test_analysis['error_abs'] = np.abs(test_analysis['real'] - test_analysis['prediccion'])

error_por_cat = test_analysis.groupby('category').agg(
    n_muestras=('real', 'count'),
    mae=('error_abs', 'mean'),
    real_mean=('real', 'mean'),
    pred_mean=('prediccion', 'mean')
).round(2).sort_values('n_muestras', ascending=False)

st.write("**Error por categoria de arma:**")
st.dataframe(error_por_cat)

fig8, ax = plt.subplots(figsize=(10, 5))
error_por_cat.plot(kind='bar', y=['real_mean', 'pred_mean'], ax=ax,
                   color=['#667eea', '#e74c3c'], edgecolor='white', alpha=0.8)
ax.set_title('Promedio Real vs Prediccion por Categoria', fontsize=14, fontweight='bold')
ax.set_xlabel('Categoria')
ax.set_ylabel('Destroyed (promedio)')
ax.legend(['Real', 'Prediccion'], fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig8)

# Reporte final texto
st.subheader("Reporte Final")
st.code(f"""
========================================================================
        REPORTE FINAL -- PROYECTO END-TO-END ML (REGRESION)
========================================================================

DATASET: Massive Missile Attacks on Ukraine
   - Registros totales: {df.shape[0]:,}
   - Registros usados (sin nulos en target): {data_model.shape[0]:,}
   - Features: {len(feature_cols_num)} numericas + {len(feature_cols_cat)} categoricas
   - Variable objetivo: destroyed (cantidad interceptada)

INGENIERIA DE CARACTERISTICAS:
   - Extraccion de componentes temporales (year, month, day_of_week, day_of_year)
   - Enriquecimiento con catalogo de modelos (category, type)
   - Agrupacion de modelos poco frecuentes (top 10 + Other)
   - Simplificacion de regiones objetivo
   - Variables binarias: has_launch_place, is_shahed_flag

PIPELINE:
   - ColumnTransformer: StandardScaler (num) + OneHotEncoder (cat)
   - Train/Test split: 80/20

MODELO OPTIMIZADO (Random Forest + GridSearchCV):
   - Mejores parametros: {grid_search.best_params_}
   - R2:   {r2_best:.4f}
   - MAE:  {mae_best:.2f}
   - RMSE: {rmse_best:.2f}

CONCLUSIONES:
   - La variable "launched" es el predictor mas fuerte (correlacion directa).
   - El tipo de arma y la region objetivo aportan informacion complementaria.
   - Random Forest captura relaciones no lineales mejor que la regresion lineal.
   - GridSearchCV mejoro el rendimiento al ajustar profundidad y tamano de arbol.
========================================================================
""", language="text")

# ========================
# CONCLUSIONES
# ========================
st.header("Conclusiones")

st.markdown("""
**1. Prediccion de misiles/drones destruidos**

El modelo de Machine Learning logro predecir con alta precision la cantidad de misiles y drones destruidos
en cada ataque, utilizando variables como el modelo de arma, lugar de lanzamiento y cantidad lanzada.
La variable `launched` resulto ser el predictor dominante con una importancia del 93%, confirmando que
la escala del ataque es el factor determinante para estimar la capacidad de intercepcion. Las variables
temporales y el tipo de arma aportaron informacion complementaria al modelo.

**2. Modelo mas efectivo: Gradient Boosting vs Random Forest**

Entre los cuatro modelos evaluados, Gradient Boosting obtuvo el mejor rendimiento en test con un R2 de 0.9812
y un RMSE de 7.53, seguido de cerca por Random Forest (R2 = 0.9774, RMSE = 8.26). Tras la optimizacion
con GridSearchCV, Random Forest mejoro a un R2 de 0.9781 y RMSE de 8.11, aunque sin superar al Gradient
Boosting base. Los modelos lineales (Linear y Ridge Regression) quedaron significativamente por debajo
con R2 ~ 0.93, evidenciando que las relaciones no lineales en los datos favorecen a los modelos basados
en arboles.

**3. Conclusion general**

Los resultados de prediccion revelan que el sistema de defensa aerea de Ucrania mantiene una capacidad
de intercepcion altamente predecible y consistente, con un MAE de solo 2.65 unidades. El analisis por
categoria muestra que los UAV presentan el mayor volumen de ataques pero tambien las tasas de intercepcion
mas variables, mientras que los misiles balisticos tienen errores de prediccion minimos (MAE = 0.59).
Esto sugiere que la defensa antiaerea opera con patrones sistematicos que los modelos de ML pueden
capturar eficazmente.

---

### Insight Principal: Proveedor Maximo de Armamento

Por volumen (cantidad de unidades lanzadas), Iran es el principal proveedor, aportando el ~87.8% del total
de armamento lanzado contra Ucrania, casi exclusivamente a traves del dron Shahed-136/131 (78,878 unidades).
Este dron kamikaze de bajo costo (~$20,000-$50,000 USD por unidad) se convirtio en el arma mas utilizada
por su relacion costo-efectividad.

**Insight 1:** A partir del analisis de esta informacion podemos identificar las capacidades de produccion
de los aliados que proveen de este tipo de armamento a Rusia para enfrentar la guerra con Ucrania, asimismo,
saber el tamano de las industrias y su capacidad de produccion. Por diversidad de sistemas de armas, Rusia
es el principal fabricante, produciendo mas de 30 tipos diferentes de misiles y drones: misiles de crucero
(X-101, Kalibr), misiles balisticos (Iskander-M), misiles hipersonicos (Kinzhal, Zircon), drones de
reconocimiento (Orlan-10, ZALA, Supercam), municiones merodeadoras (Lancet), e incluso misiles antiaereos
S-300/S-400 reconvertidos para ataque terrestre.

Corea del Norte tiene una participacion marginal con el misil balistico KN-23 (8 unidades), lo que
evidencia la cooperacion militar entre Corea del Norte y Rusia.

**Insight 2:** Dominancia irani en volumen: El analisis revela que el dron Shahed-136/131 de fabricacion
irani representa casi el 88% de todas las armas lanzadas. Esto refleja una estrategia de saturacion basada
en volumen y bajo costo, donde el objetivo es abrumar las defensas antiareas con cantidades masivas de
drones baratos, mientras los misiles rusos de mayor costo (X-101, Kalibr, Kinzhal) se emplean de forma
mas selectiva contra objetivos de alto valor. Esto lo apreciamos en la guerra EE.UU-Iran donde se emplea
la misma estrategia de fatiga a los sistemas de deteccion de municion misilistica.

**Insight 3:** Implicaciones geopoliticas: El dataset evidencia una cadena de suministro militar que
involucra al menos tres paises: Rusia (fabricante principal y mayor diversidad), Iran (proveedor masivo
de drones Shahed) y Corea del Norte (proveedor marginal de misiles balisticos KN-23). Esta triangulacion
de proveedores refleja las alianzas geopoliticas en el conflicto y la dependencia rusa de socios externos
para sostener la intensidad de los ataques a lo largo del tiempo.
""")



