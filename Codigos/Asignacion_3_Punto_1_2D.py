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
#----------------------------------------------------------------------------------------------------#

import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
import csv
import math


# Cargar los datos desde el archivo CSV con coma como delimitador
data = pd.read_csv('datosmasas.csv', delimiter=',')

# Calcular la masa total del sistema
masa_total = data['masas'].sum()

# Variable que quieres guardar en un archivo de texto
mi_variable = str(masa_total)

# Nombre del archivo donde guardarás la variable
nombre_archivo = "masatotal.txt"

# Abrir el archivo en modo escritura
with open(nombre_archivo, 'w') as archivo:
    # Escribir la variable en el archivo
    archivo.write(mi_variable)

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

# Calcular el centro de masa del sistema en coordenadas x e y
centro_de_masa_x = (data['masas'] * data['x']).sum() / masa_total
centro_de_masa_y = (data['masas'] * data['y']).sum() / masa_total

centro_de_masa = [[centro_de_masa_x, centro_de_masa_y]]

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


original = tensor_momento_inercia_2D

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

# Extraer coordenadas x e y de las partículas y sus masas
x = data['x']
y = data['y']
masas = data['masas']

# Escalar las masas para que sean visibles en la gráfica
tamaño_puntos = 100 * masas / np.max(masas)  # Escalar el tamaño de los puntos según las masas

# Primera parte: Distribución de masas en el plano xy
figura_masas = plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=tamaño_puntos, color='blue', alpha=0.5)  # Puntos azules escalados por masa
plt.scatter(centro_de_masa_x, centro_de_masa_y, marker='x', color='red', label='Centro de masa')  # Centro de masa
plt.title('Distribución de masas en el plano xy con masas escaladas y centro de masa')
plt.xlabel('Coordenada x')
plt.ylabel('Coordenada y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # Aspecto igual para ejes x y
plt.legend()
plt.savefig('Distribucionmasas.png')

#----------------------------------------------------------------------------------------------------#


# Extraer los autovectores individuales
vec1 = autovectores[:, 0]
vec2 = autovectores[:, 1]

# Calcular los puntos de intersección con los ejes x e y
interseccion_vec1_x = vec1[0] * 2  # Ejemplo de punto de intersección con el eje x (suponiendo escala)
interseccion_vec1_y = vec1[1] * 2  # Ejemplo de punto de intersección con el eje y (suponiendo escala)

interseccion_vec2_x = vec2[0] * 2  # Ejemplo de punto de intersección con el eje x (suponiendo escala)
interseccion_vec2_y = vec2[1] * 2  # Ejemplo de punto de intersección con el eje y (suponiendo escala)

# Crear la figura y los ejes para autovectores
fig, ax = plt.subplots(figsize=(6, 6))  # Tamaño reducido en un 25%

# Graficar los autovectores desde el origen (0, 0)
ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=0.75, color='b', label=f'Autovector 1: {vec1}')
ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=0.75, color='r', label=f'Autovector 2: {vec2}')

# Ajustar los límites del gráfico y aspecto
ax.set_xlim([-2, 2])  # Ajustar según tus datos
ax.set_ylim([-2, 2])  # Ajustar según tus datos
ax.set_aspect('equal')

# Configurar los ejes
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('X', size=14, labelpad=-10)  # Etiqueta y ajuste de tamaño
ax.set_ylabel('Y', size=14, labelpad=-10, rotation=0)  # Etiqueta y ajuste de tamaño

# Etiquetas y ticks de los ejes
ticks_frequency = 0.5
x_ticks = np.arange(-1.5, 1.6, ticks_frequency)
y_ticks = np.arange(-1.5, 1.6, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])

# Título del gráfico
plt.title('Autovectores', fontsize=16, pad=10)

# Rejilla
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

# Mostrar la figura
plt.legend()
plt.savefig('Autovectores.png')

#----------------------------------------------------------------------------------------------------#

# IMPRESION DE TODO

autovalores = [autovalores]

print("-----Listas con formato 4 decimal-----")
print(" ")
tensor_momento_inercia_2D = formato_lista(tensor_momento_inercia_2D)
centro_de_masa = formato_lista(centro_de_masa)
autovalores = formato_lista(autovalores)
autovectores = formato_lista(autovectores)
matriz_transformacion = formato_lista(matriz_transformacion)

print("Tensor Momento Inercia")
print(tensor_momento_inercia_2D)

print("Centro de Masa del Sistema")
print(centro_de_masa,"tipo",type(centro_de_masa))

print("Autovalores")
print(autovalores)
print("Autovectores")
print(autovectores)
print("Matriz Transformacion")
print(matriz_transformacion)
print(" ")

#----------------------------------------------------------------------------------------------------#

# LISTAS GUARDADAS COMO MATRIZ SINTAXIS LATEX EN .TXT

archivo_tensor_momento_inercia_2D = "tensor_momento_inercia_2D.txt"

archivo_centro_masa2D = "centrodemasa2D.txt"

archivo_autovalores = "autovalores.txt"
archivo_autovectores = "autovectores.txt"
archivo_M_transformacion = "matriz_transformacion.txt"

guardar_lista_latex(tensor_momento_inercia_2D, archivo_tensor_momento_inercia_2D)
guardar_lista_latex(centro_de_masa, archivo_centro_masa2D)
guardar_lista_latex(autovalores, archivo_autovalores)
guardar_lista_latex(autovectores, archivo_autovectores)
guardar_lista_latex(matriz_transformacion, archivo_M_transformacion)


#----------------------------------------------------------------------------------------------------#