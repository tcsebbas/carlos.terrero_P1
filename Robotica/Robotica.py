import numpy as np

#.flatten() sirve para pasar de multidimension a unidimension
#[:, np.newaxis] sirve para agregar una dimension

# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Declarando inputs
inputs_00 = np.array([[0], [0]]) # Inputs transpuestos
inputs_01 = np.array([[0], [1]]) # Inputs transpuestos
inputs_10 = np.array([[1], [0]]) # Inputs transpuestos
inputs_11 = np.array([[1], [1]]) # Inputs transpuestos

num_esperado_00 = np.array([[0], [0]])
num_esperado_01 = np.array([[1], [0]])
num_esperado_10 = np.array([[1], [0]])
num_esperado_11 = np.array([[0], [1]])

epochs = 5000
learning_rate = 0.25
tolerancia_error = 0.01

# Crear la clase de la red neuronal
class RedNeuronal:
    def __init__(self):
        # ___Inicializamos los pesos y bias___
        self.bias_izquierda = np.random.rand(3)     # Bias para las 3 neuronas de la izquierda
        self.pesos_izquierda = np.random.rand(3, 2) # Pesos para 3 neuronas de la izquierda
        self.bias_derecha = np.random.rand(2)       # Bias para las 2 neuronas de la derecha
        self.pesos_derecha = np.random.rand(2, 3)   # Pesos para 2 neuronas de la derecha

    def funcion(self, inputs):
        # ___Para la capa 1 de la neurona___
        sumatori_izquierda = np.dot(self.pesos_izquierda, inputs) + self.bias_izquierda[:, np.newaxis]
        izquierda_a_derecha = sigmoid(sumatori_izquierda) # Formula pdf

        # ___Para la capa 2 de la neurona___
        sumatori_derecha = np.dot(self.pesos_derecha, izquierda_a_derecha) + self.bias_derecha[:, np.newaxis]
        valor_salida = sigmoid(sumatori_derecha) # Formula pdf

        return valor_salida, izquierda_a_derecha

    def entrenamiento(self, inputs, num_esperado): # For para actualizar pesos y bias
        for epoch in range(epochs):
            valor_salida, izquierda_a_derecha = self.funcion(inputs)

            # ___Derivada de Z2 = Ver valor del error___
            error = valor_salida - num_esperado  # Error entre salida y valor esperado

            # ___Ver si el error es suficientemente pequeño___
            if np.linalg.norm(error) < tolerancia_error:
                break # Salir cuando el error sea menor que n

            # ___Actualitzem els pesos layer 2___
            t_izquierda_a_derecha = izquierda_a_derecha.T
            self.pesos_derecha = self.pesos_derecha - learning_rate * np.dot(error, t_izquierda_a_derecha) # Formula pdf

            # ___Actualitzem les bias layer 2___
            self.bias_derecha = self.bias_derecha - learning_rate * error.flatten() # Formula pdf

            # ___Deltas de layer 1___
            peso_derecha_trans = self.pesos_derecha.T
            lado_izquierdo = np.dot(peso_derecha_trans, error) # Formula pdf
            lado_derecho = np.multiply(izquierda_a_derecha, 1 - izquierda_a_derecha) # Formula pdf
            d_l1 = np.multiply(lado_izquierdo, lado_derecho) # Formula pdf

            # ___Actualitzem pesos layer 1___
            dv_peso_izq = np.dot(d_l1, inputs.T)
            self.pesos_izquierda = self.pesos_izquierda - learning_rate * dv_peso_izq # Formula pdf

            # ___Actualitzem les bias layer 1___
            self.bias_izquierda = self.bias_izquierda - learning_rate * d_l1.flatten() # Formula pdf

        return valor_salida

red = RedNeuronal()
salida_00 = red.entrenamiento(inputs_00, num_esperado_00)
salida_01 = red.entrenamiento(inputs_01, num_esperado_01)
salida_10 = red.entrenamiento(inputs_10, num_esperado_10)
salida_11 = red.entrenamiento(inputs_11, num_esperado_11)

print(" X1 X2   XOR AND")
print(num_esperado_00.T, np.round(salida_00.T, 1))
print(num_esperado_01.T, np.round(salida_01.T, 1))
print(num_esperado_10.T, np.round(salida_10.T, 1))
print(num_esperado_11.T, np.round(salida_11.T, 1))
