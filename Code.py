# =============================================================================
# Proyecto End-to-End de Machine Learning (Regresion)
# Dataset: Massive Missile Attacks on Ukraine
# Dashboard interactivo con filtros laterales
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ruta base relativa al archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# CONFIGURACION DE LA APP
# ========================
st.set_page_config(
    page_title="Missile Attacks Ukraine - ML End-to-End",
    page_icon="\U0001F680",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CARGA DE DATOS
# ========================
@st.cache_data
def cargar_datos():
    df = pd.read_csv(os.path.join(BASE_DIR, 'missile_attacks_daily.csv'))
    cat = pd.read_csv(os.path.join(BASE_DIR, 'missiles_and_uavs.csv'))
    return df, cat

df_raw, df_catalog = cargar_datos()

# Merge para obtener category, national_origin y type
df = df_raw.merge(
    df_catalog[['model', 'category', 'national_origin', 'type']],
    on='model', how='left'
)
df['category'] = df['category'].fillna('Unknown')
df['national_origin'] = df['national_origin'].fillna('unknown')
df['type'] = df['type'].fillna('unknown')

# ========================
# SIDEBAR - FILTROS
# ========================
st.sidebar.title("Filtros")
st.sidebar.markdown("---")

# Filtro por categoria de arma
categorias_disponibles = sorted(df['category'].dropna().unique().tolist())
sel_categorias = st.sidebar.multiselect(
    "Categoria de arma",
    options=categorias_disponibles,
    default=categorias_disponibles,
    help="Filtrar por tipo de arma: cruise missile, UAV, ballistic, etc."
)

# Filtro por origen nacional
origenes_disponibles = sorted(df['national_origin'].dropna().unique().tolist())
sel_origenes = st.sidebar.multiselect(
    "Origen del arma",
    options=origenes_disponibles,
    default=origenes_disponibles,
    help="Filtrar por pais de fabricacion"
)

# Filtro por modelo especifico (opcional)
modelos_disponibles = sorted(df['model'].dropna().unique().tolist())
sel_modelos = st.sidebar.multiselect(
    "Modelo especifico (opcional)",
    options=modelos_disponibles,
    default=[],
    help="Dejar vacio para incluir todos los modelos"
)

st.sidebar.markdown("---")
st.sidebar.caption("Tarea Semana 3 - Modelos Analiticos")

# Aplicar filtros
mask = df['category'].isin(sel_categorias) & df['national_origin'].isin(sel_origenes)
if sel_modelos:
    mask = mask & df['model'].isin(sel_modelos)
df_filtered = df[mask].copy()

# ========================
# TITULO PRINCIPAL
# ========================
st.markdown("""
# Tarea Semana 3 -- Proyecto End-to-End de Machine Learning
## Dataset: *Massive Missile Attacks on Ukraine*

**Objetivo:** Predecir la cantidad de misiles/drones **destruidos** (`destroyed`) en cada ataque,
a partir de las caracteristicas del ataque (modelo de arma, lugar de lanzamiento, cantidad lanzada, etc.).
""")

# Metricas resumen del filtro actual
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros filtrados", f"{len(df_filtered):,}")
c2.metric("Modelos unicos", f"{df_filtered['model'].nunique()}")
total_launched = df_filtered['launched'].sum()
total_destroyed = df_filtered['destroyed'].sum()
c3.metric("Total lanzados", f"{int(total_launched):,}")
tasa_global = total_destroyed / total_launched * 100 if total_launched > 0 else 0
c4.metric("Tasa intercepcion", f"{tasa_global:.1f}%")

# ========================
# TABS
# ========================
tab_eda, tab_modelos, tab_conclusiones = st.tabs([
    "Analisis Exploratorio (EDA)",
    "Modelos de ML",
    "Conclusiones e Insights"
])

# ================================================================
# TAB 1: EDA
# ================================================================
with tab_eda:
    st.header("Analisis Exploratorio de Datos")

    # --- Grafico 1: Volumen de ataques por origen ---
    st.subheader("Volumen de ataques por pais de origen")
    vol_origen = df_filtered.groupby('national_origin')['launched'].sum().sort_values(ascending=False)
    color_map_origen = {
        'iran': '#e74c3c',
        'russia': '#3498db',
        'north korea': '#f39c12',
        'unknown': '#95a5a6'
    }
    colores_vol = [color_map_origen.get(o, '#667eea') for o in vol_origen.index]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.barh(vol_origen.index[::-1], vol_origen.values[::-1],
                    color=colores_vol[::-1], edgecolor='white')
    ax1.set_xlabel('Total lanzados', fontsize=12)
    ax1.set_title('Volumen Total de Armas Lanzadas por Pais de Origen', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, vol_origen.values[::-1]):
        ax1.text(bar.get_width() + max(vol_origen.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2, f'{int(val):,}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- Grafico 2: Top 15 modelos coloreados por origen ---
    st.subheader("Top 15 Modelos de Armas Mas Utilizados")
    top15 = df_filtered.groupby('model')['launched'].sum().sort_values(ascending=False).head(15)
    origen_por_modelo = df_filtered.drop_duplicates('model').set_index('model')['national_origin']
    colores_top15 = [color_map_origen.get(origen_por_modelo.get(m, 'unknown'), '#667eea')
                     for m in top15.index]

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars = ax2.barh(top15.index[::-1], top15.values[::-1],
                    color=colores_top15[::-1], edgecolor='white')
    ax2.set_xlabel('Total lanzados', fontsize=12)
    ax2.set_title('Top 15 Modelos (coloreados por pais de origen)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, top15.values[::-1]):
        ax2.text(bar.get_width() + max(top15.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2, f'{int(val):,}', va='center', fontsize=9)
    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l.title())
                       for l, c in color_map_origen.items() if l in origen_por_modelo.values]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Grafico 3: Tasa de intercepcion ---
    st.subheader("Tasa de Intercepcion por Modelo (Top 10)")
    top10_names = df_filtered['model'].value_counts().head(10).index
    df_t10 = df_filtered[df_filtered['model'].isin(top10_names)].copy()
    df_t10['interception_rate'] = df_t10['destroyed'] / df_t10['launched'].replace(0, np.nan)
    tasa_modelo = df_t10.groupby('model')['interception_rate'].mean().sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    colors_tasa = ['#2ecc71' if v > 0.8 else '#e67e22' if v > 0.5 else '#e74c3c'
                   for v in tasa_modelo.values]
    bars = ax3.barh(tasa_modelo.index[::-1], tasa_modelo.values[::-1],
                    color=colors_tasa[::-1], edgecolor='white')
    ax3.set_xlabel('Tasa de Intercepcion Promedio', fontsize=12)
    ax3.set_title('Tasa de Intercepcion por Modelo (Top 10)', fontsize=14, fontweight='bold')
    ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    for bar, val in zip(bars, tasa_modelo.values[::-1]):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1%}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)

    col_a, col_b = st.columns(2)

    # --- Grafico 4: Distribucion de destroyed ---
    with col_a:
        st.subheader("Distribucion de Destroyed")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(df_filtered['destroyed'].dropna(), bins=50,
                 color='#667eea', edgecolor='white', alpha=0.8)
        mean_val = df_filtered['destroyed'].mean()
        ax4.axvline(mean_val, color='red', linestyle='--',
                    label=f'Media: {mean_val:.1f}')
        ax4.set_xlabel('Cantidad destruida')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribucion de Destroyed', fontsize=13, fontweight='bold')
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)

    # --- Grafico 5: Launched vs Destroyed ---
    with col_b:
        st.subheader("Launched vs Destroyed")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.scatter(df_filtered['launched'], df_filtered['destroyed'],
                    alpha=0.4, color='#764ba2', s=20)
        lmax = df_filtered['launched'].max() if len(df_filtered) > 0 else 1
        ax5.plot([0, lmax], [0, lmax], 'r--', alpha=0.5, label='Ideal (100%)')
        ax5.set_xlabel('Lanzados')
        ax5.set_ylabel('Destruidos')
        ax5.set_title('Launched vs Destroyed', fontsize=13, fontweight='bold')
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)

    # --- Grafico 6: Matriz de correlacion ---
    st.subheader("Matriz de Correlacion")
    cols_corr = ['launched', 'destroyed', 'not_reach_goal', 'still_attacking',
                 'is_shahed', 'num_hit_location', 'num_fall_fragment_location',
                 'turbojet', 'turbojet_destroyed']
    cols_present = [c for c in cols_corr if c in df_filtered.columns]
    corr_matrix = df_filtered[cols_present].corr()

    fig6, ax6 = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax6)
    ax6.set_title('Matriz de Correlacion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig6)

# ================================================================
# TAB 2: MODELOS DE ML
# ================================================================
with tab_modelos:
    st.header("Modelos de Regresion")
    st.info("Los modelos se entrenan con el **dataset completo** (sin filtros del sidebar) "
            "para garantizar la mejor generalizacion.")

    # --- Preparar datos para ML (siempre dataset completo) ---
    data_ml = df_raw.copy()
    data_ml['time_start'] = pd.to_datetime(data_ml['time_start'], errors='coerce')
    data_ml['year'] = data_ml['time_start'].dt.year
    data_ml['month'] = data_ml['time_start'].dt.month
    data_ml['day_of_week'] = data_ml['time_start'].dt.dayofweek
    data_ml['day_of_year'] = data_ml['time_start'].dt.dayofyear

    data_ml = data_ml.merge(df_catalog[['model', 'category', 'type']], on='model', how='left')
    data_ml['category'] = data_ml['category'].fillna('Unknown')
    data_ml['type'] = data_ml['type'].fillna('unknown')

    def simplificar_target(t):
        if pd.isna(t):
            return 'Unknown'
        t = str(t).lower()
        for key, val in [('ukraine', 'Ukraine_general'), ('south', 'South'),
                         ('east', 'East'), ('odesa', 'Odesa'), ('kherson', 'Kherson'),
                         ('kharkiv', 'Kharkiv'), ('kyiv', 'Kyiv'),
                         ('mykolaiv', 'Mykolaiv'), ('dnipro', 'Dnipro')]:
            if key in t:
                return val
        return 'Other'

    data_ml['target_region'] = data_ml['target'].apply(simplificar_target)
    top10_models = data_ml['model'].value_counts().head(10).index
    data_ml['model_group'] = data_ml['model'].apply(lambda x: x if x in top10_models else 'Other')
    data_ml['has_launch_place'] = data_ml['launch_place'].notna().astype(int)
    data_ml['is_shahed_flag'] = data_ml['is_shahed'].fillna(0).apply(lambda x: 1 if x > 0 else 0)

    data_model = data_ml.dropna(subset=['launched', 'destroyed']).copy()

    feature_cols_num = ['launched', 'year', 'month', 'day_of_week', 'day_of_year',
                        'has_launch_place', 'is_shahed_flag']
    feature_cols_cat = ['model_group', 'category', 'type', 'target_region']

    X = data_model[feature_cols_num + feature_cols_cat]
    y = data_model['destroyed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    st.write(f"**Train:** {X_train.shape[0]:,} muestras  |  **Test:** {X_test.shape[0]:,} muestras")

    # --- Entrenar modelos (cacheado) ---
    @st.cache_resource
    def entrenar_modelos(_preprocessor, _X_train, _y_train, _X_test, _y_test):
        modelos = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        resultados = []
        for nombre, modelo in modelos.items():
            pipe = Pipeline([('preprocessor', _preprocessor), ('model', modelo)])
            cv_scores = cross_val_score(pipe, _X_train, _y_train, cv=5, scoring='r2', n_jobs=-1)
            pipe.fit(_X_train, _y_train)
            y_pred = pipe.predict(_X_test)
            mae = mean_absolute_error(_y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(_y_test, y_pred))
            r2 = r2_score(_y_test, y_pred)
            resultados.append({
                'Modelo': nombre,
                'CV R2 (mean)': round(cv_scores.mean(), 4),
                'CV R2 (std)': round(cv_scores.std(), 4),
                'Test MAE': round(mae, 2),
                'Test RMSE': round(rmse, 2),
                'Test R2': round(r2, 4)
            })

        df_res = pd.DataFrame(resultados).sort_values('Test R2', ascending=False)

        pipe_rf = Pipeline([('preprocessor', _preprocessor),
                            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(pipe_rf, param_grid, cv=5, scoring='r2',
                                   n_jobs=-1, verbose=0, return_train_score=True)
        grid_search.fit(_X_train, _y_train)

        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(_X_test)
        mae_b = mean_absolute_error(_y_test, y_pred_best)
        rmse_b = np.sqrt(mean_squared_error(_y_test, y_pred_best))
        r2_b = r2_score(_y_test, y_pred_best)
        return df_res, grid_search, best_model, y_pred_best, mae_b, rmse_b, r2_b

    with st.spinner("Entrenando modelos (esto puede tardar unos minutos la primera vez)..."):
        df_resultados, grid_search, best_model, y_pred_best, mae_best, rmse_best, r2_best = \
            entrenar_modelos(preprocessor, X_train, y_train, X_test, y_test)

    # --- Ranking de modelos ---
    st.subheader("Ranking de Modelos Baseline")
    st.dataframe(df_resultados, use_container_width=True)

    fig_comp, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors_mod = ['#667eea', '#764ba2', '#2ecc71', '#e67e22']
    bars1 = axes[0].barh(df_resultados['Modelo'], df_resultados['Test R2'],
                         color=colors_mod[:len(df_resultados)], edgecolor='white')
    axes[0].set_xlabel('R2 Score')
    axes[0].set_title('Comparacion - R2 en Test', fontsize=13, fontweight='bold')
    for bar, val in zip(bars1, df_resultados['Test R2']):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=10)

    bars2 = axes[1].barh(df_resultados['Modelo'], df_resultados['Test MAE'],
                         color=colors_mod[:len(df_resultados)], edgecolor='white')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('Comparacion - MAE en Test', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, df_resultados['Test MAE']):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_comp)

    # --- GridSearchCV ---
    st.subheader("Optimizacion con GridSearchCV (Random Forest)")
    st.success("Mejores hiperparametros:")
    for param, val in grid_search.best_params_.items():
        st.write(f"  - **{param}:** {val}")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae_best:.2f}")
    m2.metric("RMSE", f"{rmse_best:.2f}")
    m3.metric("R2", f"{r2_best:.4f}")

    r2_base = df_resultados[df_resultados['Modelo'] == 'Random Forest']['Test R2'].values[0]
    st.write(f"**Mejora vs baseline:** R2 {r2_base:.4f} -> {r2_best:.4f} (Delta = {r2_best - r2_base:+.4f})")

    # --- Predicciones vs Real ---
    st.subheader("Predicciones vs Valores Reales")
    fig_pred, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_test, y_pred_best, alpha=0.4, color='#667eea', s=20,
                    edgecolor='white', linewidth=0.3)
    lim_max = max(y_test.max(), y_pred_best.max()) * 1.05
    axes[0].plot([0, lim_max], [0, lim_max], 'r--', alpha=0.7, label='Prediccion perfecta')
    axes[0].set_xlabel('Valor Real (destroyed)')
    axes[0].set_ylabel('Prediccion')
    axes[0].set_title(f'Real vs Predicho (R2={r2_best:.4f})', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(-5, lim_max)
    axes[0].set_ylim(-5, lim_max)

    residuos = y_test - y_pred_best
    axes[1].hist(residuos, bins=50, color='#764ba2', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Residuo (Real - Predicho)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'Distribucion de Residuos (MAE={mae_best:.2f})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_pred)

    # --- Feature Importance ---
    st.subheader("Importancia de Features")
    rf_model = best_model.named_steps['model']
    preprocessor_fitted = best_model.named_steps['preprocessor']
    ohe_features = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(feature_cols_cat)
    all_features = list(feature_cols_num) + list(ohe_features)
    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': all_features, 'Importancia': importances})
    feat_imp = feat_imp.sort_values('Importancia', ascending=False).head(20)

    fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
    bars = ax_imp.barh(feat_imp['Feature'][::-1], feat_imp['Importancia'][::-1],
                       color='#667eea', edgecolor='white')
    ax_imp.set_xlabel('Importancia')
    ax_imp.set_title('Top 20 Features (Random Forest Optimizado)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, feat_imp['Importancia'][::-1]):
        ax_imp.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_imp)

    # --- Error por categoria ---
    st.subheader("Error por Categoria de Arma")
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
    st.dataframe(error_por_cat, use_container_width=True)

    fig_err, ax_err = plt.subplots(figsize=(10, 5))
    error_por_cat.plot(kind='bar', y=['real_mean', 'pred_mean'], ax=ax_err,
                       color=['#667eea', '#e74c3c'], edgecolor='white', alpha=0.8)
    ax_err.set_title('Promedio Real vs Prediccion por Categoria', fontsize=14, fontweight='bold')
    ax_err.set_xlabel('Categoria')
    ax_err.set_ylabel('Destroyed (promedio)')
    ax_err.legend(['Real', 'Prediccion'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_err)

# ================================================================
# TAB 3: CONCLUSIONES E INSIGHTS
# ================================================================
with tab_conclusiones:
    st.header("Conclusiones e Insights")

    st.markdown("""
### 1. Prediccion de misiles/drones destruidos

El modelo de Machine Learning logro predecir con alta precision la cantidad de misiles y drones
destruidos en cada ataque, utilizando variables como el modelo de arma, lugar de lanzamiento y
cantidad lanzada. La variable `launched` resulto ser el predictor dominante con una importancia
del 93%, confirmando que la escala del ataque es el factor determinante para estimar la capacidad
de intercepcion. Las variables temporales y el tipo de arma aportaron informacion complementaria.

### 2. Modelo mas efectivo: Gradient Boosting vs Random Forest

Entre los cuatro modelos evaluados, Gradient Boosting obtuvo el mejor rendimiento en test con un
R2 de 0.9812 y un RMSE de 7.53, seguido de cerca por Random Forest (R2 = 0.9774, RMSE = 8.26).
Tras la optimizacion con GridSearchCV, Random Forest mejoro a un R2 de 0.9781 y RMSE de 8.11,
aunque sin superar al Gradient Boosting base. Los modelos lineales (Linear y Ridge) quedaron
significativamente por debajo con R2 aprox. 0.93, evidenciando que las relaciones no lineales
favorecen a los modelos basados en arboles.

### 3. Conclusion general

Los resultados revelan que el sistema de defensa aerea de Ucrania mantiene una capacidad de
intercepcion altamente predecible y consistente, con un MAE de solo 2.65 unidades. Los UAV
presentan el mayor volumen de ataques pero tambien las tasas de intercepcion mas variables,
mientras que los misiles balisticos tienen errores de prediccion minimos (MAE = 0.59). Esto
sugiere que la defensa antiaerea opera con patrones sistematicos que los modelos de ML pueden
capturar eficazmente.

---

### Insight Principal: Proveedor Maximo de Armamento

Por volumen (cantidad de unidades lanzadas), **Iran** es el principal proveedor, aportando el
~87.8% del total de armamento lanzado contra Ucrania, casi exclusivamente a traves del dron
**Shahed-136/131** (78,878 unidades). Este dron kamikaze de bajo costo (~$20,000-$50,000 USD
por unidad) se convirtio en el arma mas utilizada por su relacion costo-efectividad.

### Insight 1: Capacidades de produccion

A partir del analisis podemos identificar las capacidades de produccion de los aliados que
proveen de armamento a Rusia. Por diversidad de sistemas, **Rusia** es el principal fabricante,
con mas de 30 tipos: misiles de crucero (X-101, Kalibr), balisticos (Iskander-M), hipersonicos
(Kinzhal, Zircon), drones (Orlan-10, ZALA, Supercam), municiones merodeadoras (Lancet), e
incluso misiles antiaereos S-300/S-400 reconvertidos para ataque terrestre.

**Corea del Norte** tiene participacion marginal con el misil balistico KN-23 (8 unidades),
evidenciando la cooperacion militar entre Corea del Norte y Rusia.

### Insight 2: Dominancia irani en volumen

El dron Shahed-136/131 de fabricacion irani representa casi el 88% de todas las armas lanzadas.
Esto refleja una estrategia de saturacion basada en volumen y bajo costo, donde el objetivo es
abrumar las defensas antiareas con cantidades masivas de drones baratos, mientras los misiles
rusos de mayor costo (X-101, Kalibr, Kinzhal) se emplean selectivamente contra objetivos de
alto valor.

### Insight 3: Implicaciones geopoliticas

El dataset evidencia una cadena de suministro militar que involucra al menos tres paises:
**Rusia** (fabricante principal y mayor diversidad), **Iran** (proveedor masivo de drones Shahed)
y **Corea del Norte** (proveedor marginal de misiles balisticos KN-23). Esta triangulacion
refleja las alianzas geopoliticas en el conflicto y la dependencia rusa de socios externos para
sostener la intensidad de los ataques a lo largo del tiempo.
""")

    # Grafico resumen: proporcion de volumen por origen
    st.subheader("Proporcion de Armamento por Pais de Origen")
    vol_total = df.groupby('national_origin')['launched'].sum().sort_values(ascending=False)
    colores_pie = [color_map_origen.get(o, '#667eea') for o in vol_total.index]

    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax_pie.pie(
        vol_total.values, labels=[o.title() for o in vol_total.index],
        autopct='%1.1f%%', colors=colores_pie, startangle=90,
        textprops={'fontsize': 12}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax_pie.set_title('Proporcion de Armas Lanzadas por Pais de Origen',
                     fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_pie)

