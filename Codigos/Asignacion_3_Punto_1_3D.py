"""
Asignación 3

Métodos Matemáticos para Físicos I

Profesor:
Luis Nuñez

Autores:
Santiago Correa - 2182212 
Jeicor Esneider Florez Pabón - 2231338
Juan José Camacho Olmos - 2180800  

"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Cargar los datos desde el archivo CSV con coma como delimitador
data = pd.read_csv('datosmasas.csv', delimiter=',')

# Verificar las primeras filas del DataFrame para asegurarnos de que los datos se hayan cargado correctamente
print(data.head())

#----------------------------------------------------------------------------------------------------#

# Calcular el centro de masa
center_of_mass = data[['x', 'y', 'z']].mean().values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar el centro de masa
ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], c='red', marker='o', s=200, label='Centro de Masa')


# Escalar los puntos en función de sus masas
scaled_masses = data['masas'] * 10

# Graficar los puntos escalados con colores según sus masas
sc = ax.scatter(data['x'], data['y'], data['z'], c=data['masas'], s=scaled_masses, cmap='viridis', alpha=0.6)

# Graficar el centro de masa
ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], c='red', marker='o', s=200, label='Centro de Masa')

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Título y leyenda
ax.set_title('Distribución de Masas en el Espacio 3D')
ax.legend()

# Mostrar la gráfica
plt.colorbar(sc, label='Masas')
plt.show()

#----------------------------------------------------------------------------------------------------#

# Calcular la masa total del sistema
masa_total = data['masas'].sum()

# Calcular el centro de masa en 3D
centro_de_masa_3d = (data[['x', 'y', 'z']] * data['masas'].values[:, np.newaxis]).sum() / masa_total

print("Centro de Masa del sistema (coordenada x):", centro_de_masa_3d.iloc[0])
print("Centro de Masa del sistema (coordenada y):", centro_de_masa_3d.iloc[1])
print("Centro de Masa del sistema (coordenada z):", centro_de_masa_3d.iloc[2])

#----------------------------------------------------------------------------------------------------#

# Calcular el tensor de inercia en 3D
def calcular_tensor_inercia_3d(data):
    # Obtener las masas y las coordenadas
    masas = data['masas']
    coordenadas = data[['x', 'y', 'z']]

    # Calcular el centro de masa
    centro_de_masa = np.average(coordenadas, axis=0, weights=masas)

    # Centrar las coordenadas en el centro de masa
    coordenadas_centrales = coordenadas - centro_de_masa

    # Calcular el tensor de inercia
    tensor_inercia = np.zeros((3, 3))
    for i in range(len(masas)):
        tensor_inercia += masas[i] * np.outer(coordenadas_centrales.iloc[i], coordenadas_centrales.iloc[i])

    return tensor_inercia

# Calcular el tensor de inercia en 3D
tensor_inercia_3d = calcular_tensor_inercia_3d(data)
print("Tensor de inercia en 3D:")
print(tensor_inercia_3d)

#----------------------------------------------------------------------------------------------------#

# Calcular los autovectores y autovalores del tensor de inercia en 3D
autovalores, autovectores = np.linalg.eig(tensor_inercia_3d)

# Imprimir los autovalores y autovectores
print("Autovalores:")
print(autovalores)
print("\nAutovectores:")
print(autovectores)

# Definir los autovectores y autovalores
autovectores = np.array([[0.70809967, 0.7061125, 0], [-0.7061125, 0.70809967, 0], [0, 0, 1]])
autovalores = np.array([1.87284938e9, 4.93463569e7, 0])  # Agregar un autovalor nulo para la tercera dimensión

# Datos de los autovectores (puntos de inicio)
x0 = 0
y0 = 0
z0 = 0

# Colores para cada autovector
colores = ['red', 'green', 'blue']

# Configuración de la figura
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los autovectores como flechas
for i in range(len(autovalores)):
    ax.quiver(x0, y0, z0, autovectores[0, i], autovectores[1, i], autovectores[2, i],
              arrow_length_ratio=0.1, color=colores[i], label=f'Autovalor {i+1}: {autovalores[i]:.2e}')

# Marcar los puntos de intersección de los autovectores con los ejes
ax.plot([0, autovectores[0, 0]], [0, autovectores[1, 0]], [0, autovectores[2, 0]], color='blue', linestyle='--', linewidth=1, alpha=0.7)
ax.plot([0, autovectores[0, 1]], [0, autovectores[1, 1]], [0, autovectores[2, 1]], color='blue', linestyle='--', linewidth=1, alpha=0.7)

# Configuración de los ejes
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Añadir leyenda
ax.legend()

# Rejilla
ax.grid(True)

plt.show()

#----------------------------------------------------------------------------------------------------#