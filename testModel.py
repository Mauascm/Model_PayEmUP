import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score



import numpy as np

import tensorflow as tf

### analisis breve de los datos contenidos en la base de datos ### 

url = 'https://raw.githubusercontent.com/Mauascm/Model_PayEmUP/main/Salary2USACleaned.csv'
data = pd.read_csv(url)

#### tomando solamente una muestra de los datos para tener una prueba rápida.
data = data.sample(frac=0.05, random_state=42)

# Ver las primeras filas de los datos
print(data.head())

# Ver información general sobre los datos
print(data.info())

# Ver estadísticas descriptivas de las variables numéricas
print(data.describe())

# Ver la cantidad de valores únicos en cada columna
print(data.nunique())

# Ver la cantidad de valores nulos en cada columna
print(data.isnull().sum())

# Eliminar las filas con valores nulos
data = data.dropna()


### ---------------------------- ###

from sklearn.preprocessing import PolynomialFeatures

# Definir las columnas socio-demográficas y académicas
socio_demographic_cols = ['CASE_STATUS', 'EMPLOYER_NAME', 'PREVAILING_WAGE_SUBMITTED', 'PREVAILING_WAGE_SUBMITTED_UNIT', 'WORK_CITY', 'WORK_STATE', 'FULL_TIME_POSITION_Y_N', 'VISA_CLASS']
academic_cols = ['PREVAILING_WAGE_SOC_CODE', 'PREVAILING_WAGE_SOC_TITLE', 'JOB_TITLE_SUBGROUP']

# Crear los transformadores para las columnas numéricas y categóricas
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Crear un transformador polinomial
poly_transformer = PolynomialFeatures(degree=2)

# Crear un preprocesador que aplique las transformaciones a las columnas correspondientes
preprocessor_socio_demographic = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['PREVAILING_WAGE_SUBMITTED']),
        ('cat', cat_transformer, socio_demographic_cols)])

preprocessor_academic = ColumnTransformer(
    transformers=[
        ('num', num_transformer, []),
        ('cat', cat_transformer, academic_cols)])

# Crear un pipeline que aplique el preprocesador, el transformador polinomial y luego ajuste el modelo
pipeline_socio_demographic = Pipeline(steps=[('preprocessor', preprocessor_socio_demographic),
                                              ('poly', poly_transformer),
                                              ('regressor', LinearRegression())])

pipeline_academic = Pipeline(steps=[('preprocessor', preprocessor_academic),
                                    ('poly', poly_transformer),
                                    ('regressor', LinearRegression())])

# La variable objetivo es el salario pagado
salary = data['PAID_WAGE_PER_YEAR']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_socio_demographic, X_test_socio_demographic, y_train_socio_demographic, y_test_socio_demographic = train_test_split(data[socio_demographic_cols], salary, test_size=0.2, random_state=42)
X_train_academic, X_test_academic, y_train_academic, y_test_academic = train_test_split(data[academic_cols], salary, test_size=0.2, random_state=42)

# Ajustar el pipeline y predecir los salarios
pipeline_socio_demographic.fit(X_train_socio_demographic, y_train_socio_demographic)
y_pred_socio_demographic = pipeline_socio_demographic.predict(X_test_socio_demographic)

pipeline_academic.fit(X_train_academic, y_train_academic)
y_pred_academic = pipeline_academic.predict(X_test_academic)

print('Datos preprocesados y divididos en conjuntos de entrenamiento y prueba.')

# Entrenar los modelos
pipeline_socio_demographic.fit(X_train_socio_demographic, y_train_socio_demographic)
pipeline_academic.fit(X_train_academic, y_train_academic)

# Transformar los datos de prueba
X_test_socio_demographic_transformed = pipeline_socio_demographic.named_steps['preprocessor'].transform(X_test_socio_demographic)
X_test_socio_demographic_transformed = pipeline_socio_demographic.named_steps['poly'].transform(X_test_socio_demographic_transformed)

X_test_academic_transformed = pipeline_academic.named_steps['preprocessor'].transform(X_test_academic)
X_test_academic_transformed = pipeline_academic.named_steps['poly'].transform(X_test_academic_transformed)

# Predecir los salarios
y_pred_socio_demographic = pipeline_socio_demographic.named_steps['regressor'].predict(X_test_socio_demographic_transformed)
y_pred_academic = pipeline_academic.named_steps['regressor'].predict(X_test_academic_transformed)

# Calcular el error cuadrático medio
mse_socio_demographic = mean_squared_error(y_test_socio_demographic, y_pred_socio_demographic)
mse_academic = mean_squared_error(y_test_academic, y_pred_academic)

print('Error cuadrático medio para el modelo socio-demográfico:', mse_socio_demographic)
print('Error cuadrático medio para el modelo académico:', mse_academic)

# Calcular RMSE
rmse_socio_demographic = np.sqrt(mse_socio_demographic)
rmse_academic = np.sqrt(mse_academic)

# Calcular MAE
mae_socio_demographic = mean_absolute_error(y_test_socio_demographic, y_pred_socio_demographic)
mae_academic = mean_absolute_error(y_test_academic, y_pred_academic)

# Calcular R²
r2_socio_demographic = r2_score(y_test_socio_demographic, y_pred_socio_demographic)
r2_academic = r2_score(y_test_academic, y_pred_academic)

print('RMSE para el modelo socio-demográfico:', rmse_socio_demographic)
print('RMSE para el modelo académico:', rmse_academic)

print('MAE para el modelo socio-demográfico:', mae_socio_demographic)
print('MAE para el modelo académico:', mae_academic)

print('R² para el modelo socio-demográfico:', r2_socio_demographic)
print('R² para el modelo académico:', r2_academic)
