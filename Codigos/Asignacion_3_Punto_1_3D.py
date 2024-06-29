"""
Asignación 3

Métodos Matemáticos para Físicos I

Profesor:
Luis Nuñez

Autores:
Santiago Correa Vergara - 2182212 
Jeicor Esneider Florez Pabón - 2231338
Juan José Camacho Olmos - 2180800  

"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Cargar los datos desde el archivo CSV con coma como delimitador
data = pd.read_csv('datosmasas.csv', delimiter=',')

#----------------------------------------------------------------------------------------------------#

# Calcular la masa total del sistema
masa_total = data['masas'].sum()

#----------------------------------------------------------------------------------------------------#

# Formato muestra solo 4 decimas

#np.set_printoptions(precision=4)

import numpy as np

def formato_lista(lista_floats):
    formato = "{:.4f}"
    
    def formatear_elemento(elemento):
        if isinstance(elemento, float):
            return float(formato.format(elemento))
        elif isinstance(elemento, list):
            return [formatear_elemento(sub_elemento) for sub_elemento in elemento]
        elif isinstance(elemento, np.ndarray):
            return np.array([formatear_elemento(sub_elemento) for sub_elemento in elemento])
        else:
            raise ValueError("Elemento no es ni un float, lista ni numpy.ndarray")
    
    return formatear_elemento(lista_floats)

# Devuele la lista con formato 4 decimales

#----------------------------------------------------------------------------------------------------#



def convertir_a_float(lista):
    float_lista = []
    
    if isinstance(lista, np.ndarray):
        lista = lista.flatten()  # Aplanamos el array numpy si es necesario
    
    for elemento in lista:
        try:
            numero_float = float(elemento)
            float_lista.append(numero_float)
        except (ValueError, TypeError):
            continue
    
    return float_lista


#----------------------------------------------------------------------------------------------------#


# Funcion que nos sirve para guardar una lista como matriz en .txt

def guardar_lista_latex(lista_datos, nombre_archivo):
    # Abrir el archivo para escritura en modo texto, sobreescribiendo si ya existe
    with open(nombre_archivo, 'w') as f:
        # Escribir el contenido LaTeX
        f.write("\\begin{center}\n")
        f.write("    \\text{Datos} =\n")
        f.write("        \\begin{pmatrix}\n")
        
        # Escribir los datos
        for fila in lista_datos:
            f.write("        ")
            f.write(" & ".join(map(str, fila)))  # Convertir elementos a cadena y unir con ' & '
            f.write(" \\\\ \n")  # Doble barra invertida para nueva línea en LaTeX
        
        f.write("        \\end{pmatrix}\n")
        f.write("\\end{center}\n")

    # Confirmación de guardado
    print(f"Los datos se han guardado en el archivo '{nombre_archivo}'.")

#----------------------------------------------------------------------------------------------------#

# Calcular el centro de masa
centro_de_masa_3D = data[['x', 'y', 'z']].mean().values


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar el centro de masa
ax.scatter(centro_de_masa_3D[0], centro_de_masa_3D[1], centro_de_masa_3D[2], c='red', marker='o', s=200, label='Centro de Masa')

# Escalar los puntos en función de sus masas
masas_escaladas = data['masas'] * 10

# Graficar los puntos escalados con colores según sus masas
sc = ax.scatter(data['x'], data['y'], data['z'], c=data['masas'], s=masas_escaladas, cmap='viridis', alpha=0.6)

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Título y leyenda
ax.set_title('Distribución de Masas en el Espacio 3D')
ax.legend()

# Mostrar la gráfica
plt.colorbar(sc, label='Masas')

plt.savefig('centrodemasa3d.png')

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

"""
print(tensor_inercia_3d)


tensor_inercia_3d = convertir_a_float(tensor_inercia_3d)
tensor_inercia_3d = formato_lista(tensor_inercia_3d)
print(tensor_inercia_3d)

tensor_inercia_3d = np.array(tensor_inercia_3d).reshape(3, 3)
print(tensor_inercia_3d)
"""

#----------------------------------------------------------------------------------------------------#


# Calcular los autovectores y autovalores del tensor de inercia en 3D
autovalores, autovectores = np.linalg.eig(tensor_inercia_3d)

original = tensor_inercia_3d

# P autovectores
P = autovectores

# Calcular la matriz inversa de P
P_inv = np.linalg.inv(P)

# Obtener las dimensiones usando la propiedad shape
tamaño_P_inv = P_inv.shape

# Transformar A a la base de autovectores y autovalores
matriz_transformacion = np.dot(P_inv, original)
matriz_transformacion = np.dot(matriz_transformacion, P)

#----------------------------------------------------------------------------------------------------#

# Datos de los autovectores (puntos de inicio)
x0 = 0
y0 = 0
z0 = 0

vec1 = autovectores[:, 0]
vec2 = autovectores[:, 1]
vec3 = autovectores[:, 2]

# Colores para cada autovector
colores = ['red', 'green', 'blue']

# Configuración de la figura
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los autovectores desde el origen (0, 0, 0)
ax.quiver(x0, y0, z0, vec1[0], vec1[1], vec1[2], color='b', label=f'Autovector 1: {vec1}')
ax.quiver(x0, y0, z0, vec2[0], vec2[1], vec2[2], color='r', label=f'Autovector 2: {vec2}')
ax.quiver(x0, y0, z0, vec3[0], vec3[1], vec3[2], color='g', label=f'Autovector 3: {vec3}')

# Marcar los puntos de intersección de los autovectores con los ejes
ax.plot([0, autovectores[0, 0]], [0, autovectores[1, 0]], [0, autovectores[2, 0]], color='blue', linestyle='--', linewidth=1, alpha=0.7)
ax.plot([0, autovectores[0, 1]], [0, autovectores[1, 1]], [0, autovectores[2, 1]], color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.plot([0, autovectores[0, 2]], [0, autovectores[1, 2]], [0, autovectores[2, 2]], color='green', linestyle='--', linewidth=1, alpha=0.7)

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
plt.savefig('Autovectores3d.png')

#----------------------------------------------------------------------------------------------------#

# IMPRESION DE TODO

autovalores = [autovalores]

print("-----Listas con formato 4 decimal-----")
print(" ")
#tensor_inercia_3d = convertir_a_float(tensor_inercia_3d)
tensor_inercia_3d = formato_lista(tensor_inercia_3d)

centro_de_masa_3D = [centro_de_masa_3D]
centro_de_masa_3D = formato_lista(centro_de_masa_3D)

autovalores = formato_lista(autovalores)
autovectores = formato_lista(autovectores)

matriz_transformacion = formato_lista(matriz_transformacion)

print("Tensor Momento Inercia")
print(tensor_inercia_3d)

print("Centro de Masa del Sistema")
print(centro_de_masa_3D)

print("Autovalores")
print(autovalores)
print("Autovectores")
print(autovectores)
print("Matriz Transformacion")
print(matriz_transformacion)
print(" ")

#----------------------------------------------------------------------------------------------------#

# LISTAS GUARDADAS COMO MATRIZ SINTAXIS LATEX EN .TXT

archivo_tensor_momento_inercia_3D = "tensor_momento_inercia_3D.txt"

archivo_centro_masa3D = "centrodemasa3D.txt"

archivo_autovalores = "autovalores_3D.txt"
archivo_autovectores = "autovectores_3D.txt"
archivo_M_transformacion = "matriz_transformacion_3D.txt"

guardar_lista_latex(tensor_inercia_3d, archivo_tensor_momento_inercia_3D)

guardar_lista_latex(centro_de_masa_3D, archivo_centro_masa3D)

guardar_lista_latex(autovalores, archivo_autovalores)
guardar_lista_latex(autovectores, archivo_autovectores)
guardar_lista_latex(matriz_transformacion, archivo_M_transformacion)

#----------------------------------------------------------------------------------------------------#

