import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Read the dataset
url = 'https://raw.githubusercontent.com/Mauascm/Model_PayEmUP/main/Salary2USACleaned.csv'
data = pd.read_csv(url)

# Take a sample of the data for a quick test
data = data.sample(frac=0.01, random_state=42)

# Define the socio-demographic and academic columns
socio_demographic_cols = ['CASE_STATUS', 'EMPLOYER_NAME', 'PREVAILING_WAGE_SUBMITTED', 'PREVAILING_WAGE_SUBMITTED_UNIT', 'WORK_CITY', 'WORK_STATE', 'FULL_TIME_POSITION_Y_N', 'VISA_CLASS']
academic_cols = ['PREVAILING_WAGE_SOC_CODE', 'PREVAILING_WAGE_SOC_TITLE', 'JOB_TITLE_SUBGROUP']

# Create the transformers for numeric and categorical columns
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Create a preprocessor that applies the transformations to the corresponding columns
preprocessor_socio_demographic = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['PREVAILING_WAGE_SUBMITTED']),
        ('cat', cat_transformer, socio_demographic_cols)])

preprocessor_academic = ColumnTransformer(
    transformers=[
        ('num', num_transformer, []),
        ('cat', cat_transformer, academic_cols)])

# Create pipelines that apply the preprocessor, feature selection, and fit the model
pipeline_socio_demographic = Pipeline(steps=[
    ('preprocessor', preprocessor_socio_demographic),
    ('imputer', SimpleImputer(strategy='mean')),
    ('feature_selection', RFE(RandomForestRegressor(n_estimators=10, random_state=42))),
    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
])

pipeline_academic = Pipeline(steps=[
    ('preprocessor', preprocessor_academic),
    ('imputer', SimpleImputer(strategy='mean')),
    ('feature_selection', RFE(RandomForestRegressor(n_estimators=10, random_state=42))),
    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
])

# Split the data into training and test sets
X_train_socio_demographic, X_test_socio_demographic, y_train_socio_demographic, y_test_socio_demographic = train_test_split(
    data[socio_demographic_cols], data['PAID_WAGE_PER_YEAR'], test_size=0.2, random_state=42)

X_train_academic, X_test_academic, y_train_academic, y_test_academic = train_test_split(
    data[academic_cols], data['PAID_WAGE_PER_YEAR'], test_size=0.2, random_state=42)

# Fit the models
pipeline_socio_demographic.fit(X_train_socio_demographic, y_train_socio_demographic)
pipeline_academic.fit(X_train_academic, y_train_academic)

# Make predictions
y_pred_socio_demographic_train = pipeline_socio_demographic.predict(X_train_socio_demographic)
y_pred_socio_demographic_test = pipeline_socio_demographic.predict(X_test_socio_demographic)

y_pred_academic_train = pipeline_academic.predict(X_train_academic)
y_pred_academic_test = pipeline_academic.predict(X_test_academic)

# Evaluate the models using R2 score
train_score_socio_demographic = r2_score(y_train_socio_demographic, y_pred_socio_demographic_train)
test_score_socio_demographic = r2_score(y_test_socio_demographic, y_pred_socio_demographic_test)

train_score_academic = r2_score(y_train_academic, y_pred_academic_train)
test_score_academic = r2_score(y_test_academic, y_pred_academic_test)

print(f'R2 score of the socio-demographic model on the training set: {train_score_socio_demographic:.2f}')
print(f'R2 score of the socio-demographic model on the test set: {test_score_socio_demographic:.2f}')

print(f'R2 score of the academic model on the training set: {train_score_academic:.2f}')
print(f'R2 score of the academic model on the test set: {test_score_academic:.2f}')
