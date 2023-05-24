# Import the necessary libraries
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('salaryTest')

# Choose a sample from the test set
sample_index = 5  # change this to choose another sample
sample_input = X_test[sample_index]
sample_input = np.reshape(sample_input.toarray(), [1, sample_input.shape[1]])

# Predict the action using the loaded model
predicted_action = model.predict(sample_input)
print(f"La acción predicha para la muestra de prueba {sample_index} es: {np.argmax(predicted_action[0])}")

# Check the true action
true_action = y_test.iloc[sample_index]
print(f"La acción verdadera para la muestra de prueba {sample_index} es: {true_action}")
