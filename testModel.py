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

import gym
from gym import spaces
import numpy as np

# Create the environment
class JobChangeEnv(gym.Env):
    def __init__(self, data, pipeline_socio_demographic, pipeline_academic):
        super(JobChangeEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,))

        # Store the data and models
        self.data = data
        self.pipeline_socio_demographic = pipeline_socio_demographic
        self.pipeline_academic = pipeline_academic

        # Initialize the current index to the first row of the data
        self.current_index = 0
    
    def step(self, action):
        # Get the current row of data
        current_row = self.data.iloc[[self.current_index]]

        # Get the current salary
        current_salary = current_row['PAID_WAGE_PER_YEAR'].values[0]

        # Get the features for the models
        socio_demographic_features = current_row[socio_demographic_cols]
        academic_features = current_row[academic_cols]

        # Get the new salary offers from the models
        socio_demographic_offer = self.pipeline_socio_demographic.predict(socio_demographic_features)[0]
        academic_offer = self.pipeline_academic.predict(academic_features)[0]


        # Calculate the reward based on the action taken by the agent
        if action == 0:  # Stay with the current job
            reward = current_salary
        elif action == 1:  # Take the socio-demographic job offer
            reward = socio_demographic_offer
        else:  # Take the academic job offer
            reward = academic_offer

        # Update the current index
        self.current_index += 1

        # Check if we have reached the end of the data
        done = self.current_index >= len(self.data)
        
        return np.array([[current_salary, socio_demographic_offer]]), reward, done, {}


    def reset(self):
    # Reset the state of the environment to an initial state
        self.current_index = 0
        return np.array([self.data.iloc[self.current_index]['PAID_WAGE_PER_YEAR'], 0, 0])


    def render(self, mode='human'):
        pass

    import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).astype('float32')
            next_state = np.array(next_state).astype('float32')
            target = self.model.predict(np.expand_dims(state, axis=0))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(np.expand_dims(next_state, axis=0))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.train_on_batch(np.expand_dims(state, axis=0), target)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


actions = list(pipeline_academic.named_steps.keys())
state_size = (3,)
action_size = len(actions)
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

n_episodes = 1000  # Set the desired number of episodes


# Training loop
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, state_size)
    for time in range(max_steps_per_episode):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("Episode: {}/{}, score: {}".format(e + 1, n_episodes, time))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)


