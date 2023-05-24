import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import numpy as np

import tensorflow as tf

### analisis breve de los datos contenidos en la base de datos ### 

url = 'https://raw.githubusercontent.com/Mauascm/Model_PayEmUP/main/salary.csv'
data = pd.read_csv(url)

#### tomando solamente una muestra de los datos para tener una prueba rápida.
data = data.sample(frac=0.005, random_state=42)

# Ver las primeras filas de los datos
print(data.head())

# Ver información general sobre los datos
print(data.info())

# Ver estadísticas descriptivas de las variables numéricas
print(data.describe())

# Ver la cantidad de valores únicos en cada columna
print(data.nunique())


### ---------------------------- ###


# Definir las columnas numéricas y categóricas
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Crear los transformadores para las columnas numéricas y categóricas
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first')

# Crear un preprocesador que aplique las transformaciones a las columnas correspondientes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Crear un pipeline que aplique el preprocesador y luego ajuste el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Aplicar las transformaciones a los datos
data_preprocessed = pipeline.fit_transform(data.drop('salary', axis=1))

# Codificar la variable objetivo
salary_encoded = (data['salary'] == ' >50K').astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, salary_encoded, test_size=0.2, random_state=42)

print('Datos preprocesados y divididos en conjuntos de entrenamiento y prueba.')


# Crear el modelo
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'Exactitud del modelo en el conjunto de entrenamiento: {train_score:.2f}')
print(f'Exactitud del modelo en el conjunto de prueba: {test_score:.2f}')




class SalaryEnv:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        self.n_samples = X.shape[0]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        # Si la acción es 0, no aumentamos el salario, por lo que la recompensa es 0
        if action == 0:
            reward = 0
        else:
            # Si la acción es 1, aumentamos el salario y la recompensa es la diferencia entre el salario predicho y el salario actual
            predicted_salary = self.model.predict(self.X[self.current_index].reshape(1, -1))
            reward = predicted_salary - self.y[self.current_index]

        # Pasar al siguiente empleado
        self.current_index += 1
        if self.current_index >= self.n_samples:
            done = True
            next_state = self.X[0]
        else:
            done = False
            next_state = self.X[self.current_index]

        return next_state, reward, done

# Resetear los índices de y_train
y_train = y_train.reset_index(drop=True)

# Crear el entorno
env = SalaryEnv(X_train, y_train, model)


class QNetwork(tf.keras.Model):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(action_size)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Crear la red neuronal
q_network = QNetwork(action_size=2)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(action_size)
        self.optimizer = tf.keras.optimizers.legacy.Adam()  # Cambiar a la versión heredada de Adam
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def get_action(self, state):
        state = np.reshape(state.toarray(), [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state.toarray(), [1, self.state_size])
        next_state = np.reshape(next_state.toarray(), [1, self.state_size])
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            next_q_values = self.model(next_state)
            target_q_values = reward + 0.99 * np.max(next_q_values) * (1 - done)
            target_q_values = tf.stop_gradient(target_q_values)
            loss = self.loss_function(q_values[0, action:action+1], [target_q_values])

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Crear el agente
agent = DQNAgent(state_size=X_train.shape[1], action_size=2)



# Número de episodios para el entrenamiento
n_episodes = 10
print(env.n_samples)
# Bucle de entrenamiento
for episode in range(n_episodes):
    # Restablecer el entorno y obtener el estado inicial
    state = env.reset()

    # Bucle para cada paso del episodio
    for step in range(env.n_samples):
        # Elegir una acción
        action = agent.get_action(state)

        # Tomar la acción y obtener la recompensa y el siguiente estado
        next_state, reward, done = env.step(action)

        # Entrenar el agente
        agent.train(state, action, reward, next_state, done)

        # Pasar al siguiente estado
        state = next_state

        # Si el episodio ha terminado, salir del bucle
        if done:
            break

print('Entrenamiento terminado.')

try:
    # Guardar el modelo
    agent.model.save('salaryTest', save_format='tf')
    print("Modelo guardado correctamente.")
except Exception as e:
    print("Error al guardar el modelo.")
    print(e)


