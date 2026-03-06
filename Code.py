#  Tarea Semana 3 — Proyecto End-to-End de Machine Learning (Regresión)
## Dataset: *Massive Missile Attacks on Ukraine*

**Objetivo:** Predecir la cantidad de misiles/drones **destruidos** (`destroyed`) en cada ataque, a partir de las características del ataque (modelo de arma, lugar de lanzamiento, cantidad lanzada, etc.).

### Pipeline completo:
1. Importación del dataset
2. Análisis exploratorio (EDA)
3. Ingeniería de características
4. Pipeline con `ColumnTransformer`
5. Modelos de regresión → Random Forest + GridSearchCV
6. Reporte final + visualizaciones
#  Tarea Semana 3 — Proyecto End-to-End de Machine Learning (Regresión)
## Dataset: *Massive Missile Attacks on Ukraine*

**Objetivo:** Predecir la cantidad de misiles/drones **destruidos** (`destroyed`) en cada ataque, a partir de las características del ataque (modelo de arma, lugar de lanzamiento, cantidad lanzada, etc.).

### Pipeline completo:
1. Importación del dataset
2. Análisis exploratorio (EDA)
3. Ingeniería de características
4. Pipeline con `ColumnTransformer`
5. Modelos de regresión → Random Forest + GridSearchCV
6. Reporte final + visualizaciones
## 1. Importación de Librerías y Datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print(' Librerías cargadas correctamente')
# Cargar datasets
df = pd.read_csv('data tarea 3/missile_attacks_daily.csv')
df_models = pd.read_csv('data tarea 3/missiles_and_uavs.csv')

print(f'Dataset principal: {df.shape[0]} filas x {df.shape[1]} columnas')
print(f'Catálogo de modelos: {df_models.shape[0]} filas x {df_models.shape[1]} columnas')
print()
df.head()
## 2. Análisis Exploratorio de Datos (EDA)
# Información general del dataset
print('='*60)
print('INFORMACIÓN GENERAL DEL DATASET')
print('='*60)
print(f'Filas: {df.shape[0]:,}')
print(f'Columnas: {df.shape[1]}')
print(f'Rango temporal: {df["time_start"].min()} a {df["time_start"].max()}')
print()
print('Tipos de datos:')
print(df.dtypes)
print()
print('Valores nulos por columna:')
nulls = df.isnull().sum()
nulls_pct = (df.isnull().sum() / len(df) * 100).round(1)
null_df = pd.DataFrame({'Nulos': nulls, '% Nulo': nulls_pct})
print(null_df[null_df['Nulos'] > 0].sort_values('% Nulo', ascending=False))
# Estadísticas descriptivas de variables numéricas clave
cols_num = ['launched', 'destroyed', 'not_reach_goal', 'still_attacking']
df[cols_num].describe().round(2)
# Distribución de la variable objetivo: destroyed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histograma
axes[0].hist(df['destroyed'].dropna(), bins=50, color='#667eea', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribución de Destroyed', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cantidad destruida')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['destroyed'].mean(), color='red', linestyle='--', label=f'Media: {df["destroyed"].mean():.1f}')
axes[0].legend()

# Boxplot
axes[1].boxplot(df['destroyed'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='#667eea', alpha=0.6))
axes[1].set_title('Boxplot de Destroyed', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Cantidad destruida')

# Relación launched vs destroyed
axes[2].scatter(df['launched'], df['destroyed'], alpha=0.4, color='#764ba2', s=20)
axes[2].plot([0, df['launched'].max()], [0, df['launched'].max()], 'r--', alpha=0.5, label='Ideal (100%)')
axes[2].set_title('Launched vs Destroyed', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Lanzados')
axes[2].set_ylabel('Destruidos')
axes[2].legend()

plt.tight_layout()
plt.show()
# Pairplot - Relaciones entre variables clave
sns.pairplot(df, vars=['launched', 'destroyed', 'not_reach_goal', 'still_attacking'],
             kind='scatter', plot_kws={'alpha': 0.5, 'color': '#667eea'})
plt.suptitle('Relaciones entre variables clave', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# Top 15 modelos más usados
top_modelos = df['model'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_modelos.index[::-1], top_modelos.values[::-1], color='#667eea', edgecolor='white')
ax.set_xlabel('Frecuencia de ataques', fontsize=12)
ax.set_title('Top 15 Modelos de Armas Más Utilizados', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top_modelos.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
plt.tight_layout()
plt.show()
# Tasa de intercepción por modelo (top 10 más usados)
top10 = df['model'].value_counts().head(10).index
df_top10 = df[df['model'].isin(top10)].copy()
df_top10['interception_rate'] = df_top10['destroyed'] / df_top10['launched']

tasa_modelo = df_top10.groupby('model')['interception_rate'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2ecc71' if v > 0.8 else '#e67e22' if v > 0.5 else '#e74c3c' for v in tasa_modelo.values]
bars = ax.barh(tasa_modelo.index[::-1], tasa_modelo.values[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Tasa de Intercepción Promedio', fontsize=12)
ax.set_title('Tasa de Intercepción por Modelo de Arma (Top 10)', fontsize=14, fontweight='bold')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
for bar, val in zip(bars, tasa_modelo.values[::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center', fontsize=9)
plt.tight_layout()
plt.show()
# Matriz de correlación de variables numéricas
cols_corr = ['launched', 'destroyed', 'not_reach_goal', 'still_attacking',
             'is_shahed', 'num_hit_location', 'num_fall_fragment_location', 'turbojet', 'turbojet_destroyed']
corr_matrix = df[cols_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
## 3. Ingeniería de Características
# Crear copia de trabajo
data = df.copy()

# 1. Parsear fechas y extraer componentes temporales
data['time_start'] = pd.to_datetime(data['time_start'], errors='coerce')
data['year'] = data['time_start'].dt.year
data['month'] = data['time_start'].dt.month
data['day_of_week'] = data['time_start'].dt.dayofweek  # 0=Lunes
data['day_of_year'] = data['time_start'].dt.dayofyear

# 2. Enriquecer con catálogo de modelos (categoría: misil, UAV, etc.)
data = data.merge(df_models[['model', 'category', 'type']], on='model', how='left')
data['category'] = data['category'].fillna('Unknown')
data['type'] = data['type'].fillna('unknown')

# 3. Simplificar target a regiones principales
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

# 4. Simplificar modelo a los top 10 + 'Other'
top10_models = data['model'].value_counts().head(10).index
data['model_group'] = data['model'].apply(lambda x: x if x in top10_models else 'Other')

# 5. Variable: tiene lugar de lanzamiento conocido
data['has_launch_place'] = data['launch_place'].notna().astype(int)

# 6. Variable is_shahed binaria
data['is_shahed_flag'] = data['is_shahed'].fillna(0).apply(lambda x: 1 if x > 0 else 0)

print(f'Dataset enriquecido: {data.shape}')
print()
print('Nuevas columnas creadas:')
print('  - year, month, day_of_week, day_of_year')
print('  - category, type (del catálogo de modelos)')
print('  - target_region (región simplificada)')
print('  - model_group (top 10 + Other)')
print('  - has_launch_place, is_shahed_flag')
data[['model', 'model_group', 'category', 'type', 'target_region', 'launched', 'destroyed']].head(10)
# Seleccionar features y target para el modelo
# Variable objetivo: destroyed
# Eliminamos filas sin 'launched' o 'destroyed'
data_model = data.dropna(subset=['launched', 'destroyed']).copy()

# Features seleccionadas
feature_cols_num = ['launched', 'year', 'month', 'day_of_week', 'day_of_year',
                    'has_launch_place', 'is_shahed_flag']
feature_cols_cat = ['model_group', 'category', 'type', 'target_region']

X = data_model[feature_cols_num + feature_cols_cat]
y = data_model['destroyed']

print(f'Tamaño de X: {X.shape}')
print(f'Tamaño de y: {y.shape}')
print(f'\nFeatures numéricas ({len(feature_cols_num)}): {feature_cols_num}')
print(f'Features categóricas ({len(feature_cols_cat)}): {feature_cols_cat}')
print(f'\nDistribución de y (destroyed):')
print(y.describe().round(2))
## 4. Pipeline con ColumnTransformer
from sklearn.impute import SimpleImputer

# Dividir en train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Train: {X_train.shape[0]:,} muestras')
print(f'Test:  {X_test.shape[0]:,} muestras')

# Definir transformadores con imputación
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

print('\n ColumnTransformer definido:')
print('  - Numéricas → SimpleImputer(median) + StandardScaler')
print('  - Categóricas → SimpleImputer(constant) + OneHotEncoder')
## 5. Modelos de Regresión

Entrenamos múltiples modelos como baseline y luego optimizamos el mejor con `GridSearchCV`.
# === Baseline: Comparar múltiples modelos ===
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

resultados = []

for nombre, modelo in modelos.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', modelo)
    ])
    
    # Cross-validation en train
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    # Entrenar y evaluar en test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    resultados.append({
        'Modelo': nombre,
        'CV R² (mean)': cv_scores.mean(),
        'CV R² (std)': cv_scores.std(),
        'Test MAE': mae,
        'Test RMSE': rmse,
        'Test R²': r2
    })
    print(f'{nombre:25s} | CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test R²: {r2:.4f} | MAE: {mae:.2f}')

df_resultados = pd.DataFrame(resultados).sort_values('Test R²', ascending=False)
print('\n' + '='*80)
print('RANKING DE MODELOS:')
print('='*80)
df_resultados
# Visualizar comparación de modelos
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# R² Score
colors = ['#667eea', '#764ba2', '#2ecc71', '#e67e22']
bars = axes[0].barh(df_resultados['Modelo'], df_resultados['Test R²'], color=colors[:len(df_resultados)], edgecolor='white')
axes[0].set_xlabel('R² Score', fontsize=12)
axes[0].set_title('Comparación de Modelos — R² en Test', fontsize=14, fontweight='bold')
for bar, val in zip(bars, df_resultados['Test R²']):
    axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)

# MAE
bars2 = axes[1].barh(df_resultados['Modelo'], df_resultados['Test MAE'], color=colors[:len(df_resultados)], edgecolor='white')
axes[1].set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
axes[1].set_title('Comparación de Modelos — MAE en Test', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, df_resultados['Test MAE']):
    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()
### 5.1 Optimización con GridSearchCV (Random Forest)
# GridSearchCV para optimizar Random Forest
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

print('Ejecutando GridSearchCV (esto puede tardar unos minutos)...')
print(f'Combinaciones a evaluar: {3*3*3*3} x 5 folds = {3*3*3*3*5} ajustes')

grid_search = GridSearchCV(
    pipe_rf, param_grid,
    cv=5, scoring='r2',
    n_jobs=-1, verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f'\n✅ Mejores hiperparámetros:')
for param, val in grid_search.best_params_.items():
    print(f'  {param}: {val}')
print(f'\nMejor CV R²: {grid_search.best_score_:.4f}')
# Evaluar modelo optimizado en test
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print('='*60)
print('RESULTADOS DEL MODELO OPTIMIZADO (Random Forest)')
print('='*60)
print(f'  MAE:  {mae_best:.2f}')
print(f'  RMSE: {rmse_best:.2f}')
print(f'  R²:   {r2_best:.4f}')
print(f'\nMejora vs baseline Random Forest:')
r2_base = df_resultados[df_resultados['Modelo']=='Random Forest']['Test R²'].values[0]
print(f'  R² baseline: {r2_base:.4f} → R² optimizado: {r2_best:.4f} (Δ = {r2_best - r2_base:+.4f})')
## 6. Reporte Final + Visualizaciones
# Predicciones vs Valores Reales
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter: Real vs Predicho
axes[0].scatter(y_test, y_pred_best, alpha=0.4, color='#667eea', s=20, edgecolor='white', linewidth=0.3)
lim_max = max(y_test.max(), y_pred_best.max()) * 1.05
axes[0].plot([0, lim_max], [0, lim_max], 'r--', alpha=0.7, label='Predicción perfecta')
axes[0].set_xlabel('Valor Real (destroyed)', fontsize=12)
axes[0].set_ylabel('Predicción', fontsize=12)
axes[0].set_title(f'Real vs Predicho (R²={r2_best:.4f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim(-5, lim_max)
axes[0].set_ylim(-5, lim_max)

# Distribución de residuos
residuos = y_test - y_pred_best
axes[1].hist(residuos, bins=50, color='#764ba2', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Residuo (Real - Predicho)', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title(f'Distribución de Residuos (MAE={mae_best:.2f})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
# Importancia de Features
rf_model = best_model.named_steps['model']
preprocessor_fitted = best_model.named_steps['preprocessor']

# Obtener nombres de features después del OneHotEncoding
ohe_features = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(feature_cols_cat)
all_features = list(feature_cols_num) + list(ohe_features)

importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': all_features, 'Importancia': importances})
feat_imp = feat_imp.sort_values('Importancia', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(feat_imp['Feature'][::-1], feat_imp['Importancia'][::-1],
               color='#667eea', edgecolor='white')
ax.set_xlabel('Importancia', fontsize=12)
ax.set_title('Top 20 Features Más Importantes (Random Forest Optimizado)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, feat_imp['Importancia'][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()
# Análisis de error por categoría de arma
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

print('Error por categoría de arma:')
print(error_por_cat)

fig, ax = plt.subplots(figsize=(10, 5))
error_por_cat.plot(kind='bar', y=['real_mean', 'pred_mean'], ax=ax,
                   color=['#667eea', '#e74c3c'], edgecolor='white', alpha=0.8)
ax.set_title('Promedio Real vs Predicción por Categoría', fontsize=14, fontweight='bold')
ax.set_xlabel('Categoría')
ax.set_ylabel('Destroyed (promedio)')
ax.legend(['Real', 'Predicción'], fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# =============================================
# REPORTE FINAL
# =============================================
print('='*70)
print('        REPORTE FINAL — PROYECTO END-TO-END ML (REGRESIÓN)')
print('='*70)
print()
print('DATASET: Massive Missile Attacks on Ukraine')
print(f'   • Registros totales: {df.shape[0]:,}')
print(f'   • Registros usados (sin nulos en target): {data_model.shape[0]:,}')
print(f'   • Features: {len(feature_cols_num)} numéricas + {len(feature_cols_cat)} categóricas')
print(f'   • Variable objetivo: destroyed (cantidad interceptada)')
print()
print('INGENIERÍA DE CARACTERÍSTICAS:')
print('   • Extracción de componentes temporales (year, month, day_of_week, day_of_year)')
print('   • Enriquecimiento con catálogo de modelos (category, type)')
print('   • Agrupación de modelos poco frecuentes (top 10 + Other)')
print('   • Simplificación de regiones objetivo')
print('   • Variables binarias: has_launch_place, is_shahed_flag')
print()
print('PIPELINE:')
print('   • ColumnTransformer: StandardScaler (num) + OneHotEncoder (cat)')
print('   • Train/Test split: 80/20')
print()
print('RESULTADOS DE MODELOS BASELINE:')
for _, row in df_resultados.iterrows():
    print(f'   {row["Modelo"]:25s} | R²: {row["Test R²"]:.4f} | MAE: {row["Test MAE"]:.2f}')
print()
print('MODELO OPTIMIZADO (Random Forest + GridSearchCV):')
print(f'   • Mejores parámetros: {grid_search.best_params_}')
print(f'   • R²:   {r2_best:.4f}')
print(f'   • MAE:  {mae_best:.2f}')
print(f'   • RMSE: {rmse_best:.2f}')
print()
print('CONCLUSIONES:')
print('   • La variable "launched" es el predictor más fuerte (correlación directa).')
print('   • El tipo de arma y la región objetivo aportan información complementaria.')
print('   • Random Forest captura relaciones no lineales mejor que la regresión lineal.')
print('   • GridSearchCV mejoró el rendimiento al ajustar profundidad y tamaño de árbol.')
print('='*70)
## Conclusiones

**1. Predicción de misiles/drones destruidos**
El modelo de Machine Learning logró predecir con alta precisión la cantidad de misiles y drones destruidos en cada ataque, utilizando variables como el modelo de arma, lugar de lanzamiento y cantidad lanzada. La variable `launched` resultó ser el predictor dominante con una importancia del 93%, confirmando que la escala del ataque es el factor determinante para estimar la capacidad de intercepción. Las variables temporales y el tipo de arma aportaron información complementaria al modelo.

**2. Modelo más efectivo: Gradient Boosting vs Random Forest**
Entre los cuatro modelos evaluados, Gradient Boosting obtuvo el mejor rendimiento en test con un R² de 0.9812 y un RMSE de 7.53, seguido de cerca por Random Forest (R² = 0.9774, RMSE = 8.26). Tras la optimización con GridSearchCV, Random Forest mejoró a un R² de 0.9781 y RMSE de 8.11, aunque sin superar al Gradient Boosting base. Los modelos lineales (Linear y Ridge Regression) quedaron significativamente por debajo con R² ≈ 0.93, evidenciando que las relaciones no lineales en los datos favorecen a los modelos basados en árboles.

**3. Conclusión general**
Los resultados de predicción revelan que el sistema de defensa aérea de Ucrania mantiene una capacidad de intercepción altamente predecible y consistente, con un MAE de solo 2.65 unidades. El análisis por categoría muestra que los UAV presentan el mayor volumen de ataques pero también las tasas de intercepción más variables, mientras que los misiles balísticos tienen errores de predicción mínimos (MAE = 0.59). Esto sugiere que la defensa antiaérea opera con patrones sistemáticos que los modelos de ML pueden capturar eficazmente.
. Insight Principal: Proveedor Máximo de Armamento
Por volumen (cantidad de unidades lanzadas), Irán es el principal proveedor, aportando el ~87.8% del total de armamento lanzado contra Ucrania, casi exclusivamente a través del dron Shahed-136/131 (78,878 unidades). Este dron kamikaze de bajo costo (~$20,000-$50,000 USD por unidad) se convirtió en el arma más utilizada por su relación costo-efectividad.

**Insight 1:** a partir del análisis de  esta información podemos identificar las capacidades de producción de los aliados que proveen del este tipo de armamento a Rusia para enfrentar la guerra con Ukrania, asimismo, saber el tamaño de las industrias y su capacidad de producción. Por diversidad de sistemas de armas, Rusia es el principal fabricante, produciendo más de 30 tipos diferentes de misiles y drones: misiles de crucero (X-101, Kalibr), misiles balísticos (Iskander-M), misiles hipersónicos (Kinzhal, Zircon), drones de reconocimiento (Orlan-10, ZALA, Supercam), municiones merodeadoras (Lancet), e incluso misiles antiaéreos S-300/S-400 reconvertidos para ataque terrestre.

Corea del Norte tiene una participación marginal con el misil balístico KN-23 (8 unidades), lo que evidencia la cooperación militar entre Corea del Norte y Rusia.

**Insight 2:** dominancia iraní en volumen: El análisis revela que el dron Shahed-136/131 de fabricación iraní representa casi el 88% de todas las armas lanzadas. Esto refleja una estrategia de saturación basada en volumen y bajo costo, donde el objetivo es abrumar las defensas antiaéreas con cantidades masivas de drones baratos, mientras los misiles rusos de mayor costo (X-101, Kalibr, Kinzhal) se emplean de forma más selectiva contra objetivos de alto valor. Esto lo apreciamos en la guerra EE.UU Iran don se emplea la misma estrategia de fatiga a los sistemas de detección de munición misilistica.

**Insight 3:** implicaciones geopolíticas: El dataset evidencia una cadena de suministro militar que involucra al menos tres países: Rusia (fabricante principal y mayor diversidad), Irán (proveedor masivo de drones Shahed) y Corea del Norte (proveedor marginal de misiles balísticos KN-23). Esta triangulación de proveedores refleja las alianzas geopolíticas en el conflicto y la dependencia rusa de socios externos para sostener la intensidad de los ataques a lo largo del tiempo.

