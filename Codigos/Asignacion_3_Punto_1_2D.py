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
import numpy as np

# Cargar los datos desde el archivo CSV con coma como delimitador
data = pd.read_csv('datosmasas.csv', delimiter=',')

# Verificar las primeras filas del DataFrame para asegurarnos de que los datos se hayan cargado correctamente
print(data.head())

# Calcular la masa total del sistema
masa_total = data['masas'].sum()

#----------------------------------------------------------------------------------------------------#

# Calcular el centro de masa del sistema en coordenadas x e y
centro_de_masa_x = (data['masas'] * data['x']).sum() / masa_total
centro_de_masa_y = (data['masas'] * data['y']).sum() / masa_total

print("Masa total del sistema:", masa_total)
print("Centro de masa del sistema (coordenada x):", centro_de_masa_x)
print("Centro de masa del sistema (coordenada y):", centro_de_masa_y)

# Calcular las coordenadas relativas al centro de masa
data['x_cm'] = data['x'] - centro_de_masa_x
data['y_cm'] = data['y'] - centro_de_masa_y

# Calcular los elementos de la matriz del tensor momento de inercia en 2D
Ixx = (data['masas'] * data['y_cm']**2).sum()
Iyy = (data['masas'] * data['x_cm']**2).sum()
Ixy = -(data['masas'] * data['x_cm'] * data['y_cm']).sum()

#----------------------------------------------------------------------------------------------------#

# Construir la matriz del tensor momento de inercia
tensor_momento_inercia_2D = np.array([[Ixx, Ixy], [Ixy, Iyy]])

# Calcular los autovalores y autovectores del tensor momento de inercia
autovalores, autovectores = np.linalg.eig(tensor_momento_inercia_2D)

print("\nCálculo del tensor momento de inercia en 2D")
print("Tensor momento de inercia en 2D:")
print(tensor_momento_inercia_2D)
print("\nAutovalores:")
print(autovalores)
print("\nAutovectores:")
print(autovectores)

#----------------------------------------------------------------------------------------------------#

# Extraer coordenadas x e y de las partículas y sus masas
x = data['x']
y = data['y']
masas = data['masas']

# Calcular el centro de masa
centro_x = np.sum(x * masas) / np.sum(masas)
centro_y = np.sum(y * masas) / np.sum(masas)

# Escalar las masas para que sean visibles en la gráfica
tamaño_puntos = 100 * masas / np.max(masas)  # Escalar el tamaño de los puntos según las masas

# Graficar distribución de masas en el plano xy con puntos escalados y centro de masa
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=tamaño_puntos, color='blue', alpha=0.5)  # Puntos azules escalados por masa
plt.scatter(centro_x, centro_y, marker='x', color='red', label='Centro de masa')  # Centro de masa
plt.title('Distribución de masas en el plano xy con masas escaladas y centro de masa')
plt.xlabel('Coordenada x')
plt.ylabel('Coordenada y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # Aspecto igual para ejes x y y
plt.legend()
plt.show()

#----------------------------------------------------------------------------------------------------#

# Definir los autovectores y autovalores
autovectores = np.array([[0.70809967, 0.7061125], [-0.7061125, 0.70809967]])
autovalores = np.array([1.87284938e9, 4.93463569e7])

# Datos de los autovectores (puntos de inicio)
x0 = 0
y0 = 0

# Colores para cada autovector
colores = ['red', 'green']

# Configuración de la figura
fig, ax = plt.subplots(figsize=(8, 8))

# Graficar los autovectores como flechas
for i in range(len(autovalores)):
    ax.quiver(x0, y0, autovectores[0, i], autovectores[1, i], angles='xy', scale_units='xy', scale=1, color=colores[i], label=f'Autovalor {i+1}: {autovalores[i]:.2e}')

# Configuración de los ejes
ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect='equal')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

# Marcar los puntos de intersección de los autovectores con los ejes
for i in range(len(autovalores)):
    ax.plot([0, autovectores[0, i]], [autovectores[1, i], autovectores[1, i]], color=colores[i], linestyle='--', linewidth=1, alpha=0.7)
    ax.plot([autovectores[0, i], autovectores[0, i]], [0, autovectores[1, i]], color=colores[i], linestyle='--', linewidth=1, alpha=0.7)

# Etiquetas de los ejes
ticks_frequency = 0.2
x_ticks = np.arange(-1.2, 1.3, ticks_frequency)
y_ticks = np.arange(-1.2, 1.3, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])

# Rejilla
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

# Leyenda
ax.legend()

plt.show()

#----------------------------------------------------------------------------------------------------#