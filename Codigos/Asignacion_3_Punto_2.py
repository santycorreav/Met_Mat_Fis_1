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
import string
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

#----------------------------------------------------------------------------------------------------#

# Nombre del archivo CSV
archivo_csv = 'gastos.csv'

# Lista para almacenar todas las filas como listas
filas = []

# Abrir el archivo CSV en modo lectura
with open(archivo_csv, 'r', newline='', encoding='utf-8') as csv_file:
    # Crear un objeto lector CSV
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    # Iterar sobre cada fila en el archivo CSV
    for fila in csv_reader:
        # Agregar la fila actual como lista a la lista de filas
        filas.append(fila)

#----------------------------------------------------------------------------------------------------#

# Sacamos de filas, cada información que necesitamos 
años = filas[0]
años = [años[0], años[2]] + años[-16:]
salud = filas[1]
salud = [salud[0], salud[2]] + salud[-16:]
investigacion = filas[2]
investigacion = [investigacion[0], investigacion[2]] + investigacion[-16:]
militar = filas[3]
militar = [militar[0], militar[2]] + militar[-16:]
educacion = filas[4]
educacion = [educacion[0], educacion[2]] + educacion[-16:]





#print(años)

#----------------------------------------------------------------------------------------------------#

# Creamos funcion que nos convierta a float

def convertir_a_float(lista):
    float_lista = []
    for elemento in lista:
        try:
            numero_float = float(elemento)
            float_lista.append(numero_float)
        except ValueError:
            continue
    return float_lista

float_años = convertir_a_float(años)
float_salud = convertir_a_float(salud)
float_investigacion = convertir_a_float(investigacion)
float_militar = convertir_a_float(militar)
float_educacion = convertir_a_float(educacion)


# Matriz de datos en float

float_datos = [[float_salud],
               [float_investigacion],
               [float_militar],
               [float_educacion]
]

"""

print("años:",float_años)
print("salud",float_salud)
print("investigacion",float_investigacion)
print("militar",float_militar)
print("educacion:",float_educacion)

"""

#----------------------------------------------------------------------------------------------------#

# Plot de todo
plt.figure(num = 'Gŕaficas')
plt.title("PIB gastado en Colombia",color='blue',fontsize = '18')
plt.xlabel("Años",color='red',fontsize = '14')
plt.ylabel("Dinero invertido",color='black',fontsize = '14')
plt.grid()
plt.plot(float_años, float_salud,'go',label='(i)Salud')
plt.plot(float_años, float_investigacion,'yo',label='(ii)Investigacion')
plt.plot(float_años, float_militar,'mo',label='(iii)Militar')
plt.plot(float_años, float_educacion,'co',label='(iv)Educacion')
plt.legend()
plt.savefig('graficas_PIB_colombia.png')

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                              INCISO A.1)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

# Calcular las medias de Salud, Investigacion, Militar y Educacion
media_salud = sum(float_salud) / len(float_salud)
media_investigacion = sum(float_investigacion) / len(float_investigacion)
media_militar = sum(float_militar) / len(float_militar)
media_educacion = sum(float_educacion) / len(float_educacion)

#----------------------------------------------------------------------------------------------------#

# Calculamos primero las varianzas de Salud (Var(S)), Investigacion (Var(I)), Militar (Var(M)) y Educacion (Var(E))

# Calcular la covarianza entre Salud y Salud (Varianza de Salud, Var(S) )
cov_S = sum((float_salud[i] - media_salud) * (float_salud[i] - media_salud) for i in range(len(float_salud))) / (len(float_salud) - 1)
#print("La covarianza entre Salud y Salud, o varianza de Salud es:", cov_S)

# Calcular la covarianza entre Investigacion e Investigacion (Varianza de Salud, Var(I) )
cov_I = sum((float_investigacion[i] - media_investigacion) * (float_investigacion[i] - media_investigacion) for i in range(len(float_investigacion))) / (len(float_investigacion) - 1)
#print("La covarianza entre Investigacion e Investigacion, o varianza de Investigacion es:", cov_I)

# Calcular la covarianza entre Militar y Militar (Varianza de Salud, Var(M) )
cov_M = sum((float_militar[i] - media_militar) * (float_militar[i] - media_militar) for i in range(len(float_militar))) / (len(float_militar) - 1)
#print("La covarianza entre Militar y Militar, o varianza de Militar es:", cov_M)

# Calcular la covarianza entre Educacion y Educacion (Varianza de Salud, Var(E) )
cov_E = sum((float_educacion[i] - media_educacion) * (float_educacion[i] - media_educacion) for i in range(len(float_educacion))) / (len(float_educacion) - 1)
#print("La covarianza entre Educacion y Educacion, o varianza de Educacion es:", cov_E)

#print(" ")

#----------------------------------------------------------------------------------------------------#

#  Las covarianzas entre S/I, S/M, S/E 

# Calcular la covarianza entre Salud e Investigacion (S/I)
cov_S_I = sum((float_salud[i] - media_salud) * (float_investigacion[i] - media_investigacion) for i in range(len(float_salud))) / (len(float_salud) - 1)
#print("La covarianza entre Salud e Investigacion es:", cov_S_I)
cov_I_S = cov_S_I

# Calcular la covarianza entre Salud y Militar (S/M)
cov_S_M = sum((float_salud[i] - media_salud) * (float_militar[i] - media_militar) for i in range(len(float_salud))) / (len(float_salud) - 1)
#print("La covarianza entre Salud y Militar es:", cov_S_M)
cov_M_S = cov_S_M

# Calcular la covarianza entre Salud y Educacion (S/E)
cov_S_E = sum((float_salud[i] - media_salud) * (float_educacion[i] - media_educacion) for i in range(len(float_salud))) / (len(float_salud) - 1)
#print("La covarianza entre Salud y Educacion es:", cov_S_E)
cov_E_S = cov_S_E

#print(" ")

#----------------------------------------------------------------------------------------------------#

# Las covarianzas entre M/I, E/I, E/M

# Calcular la covarianza entre Militar e Investigacion (M/I)
cov_M_I = sum((float_militar[i] - media_militar) * (float_investigacion[i] - media_investigacion) for i in range(len(float_militar))) / (len(float_militar) - 1)
#print("La covarianza entre Militar e Investigacion:", cov_M_I)
cov_I_M = cov_M_I

# Calcular la covarianza entre Educacion e Investigacion (E/I)
cov_E_I = sum((float_educacion[i] - media_educacion) * (float_investigacion[i] - media_investigacion) for i in range(len(float_educacion))) / (len(float_educacion) - 1)
#print("La covarianza entre Educacion e Investigacion:", cov_E_I)
cov_I_E = cov_E_I

# Calcular la covarianza entre Educacion y Militar (E/M)
cov_E_M = sum((float_educacion[i] - media_educacion) * (float_militar[i] - media_militar) for i in range(len(float_educacion))) / (len(float_educacion) - 1)
#print("La covarianza entre Educacion y Militar:", cov_E_M)
cov_M_E = cov_E_M

#print(" ")

#----------------------------------------------------------------------------------------------------#

# Entonces la matriz de covarianza es

Matriz_Covarianza = [
                    [cov_S, cov_S_I, cov_S_M, cov_S_E],
                    [cov_I_S, cov_I, cov_I_M, cov_I_E],
                    [cov_M_S, cov_M_I, cov_M, cov_M_E],
                    [cov_E_S, cov_E_I, cov_E_M, cov_E]
                        ]

print("Esta es la matriz de Covarianza:")
# Usar un bucle para imprimir cada fila en una nueva línea
for k in Matriz_Covarianza:
    print(k)

"""

# Comparación de función covarianza creada con la integrada en Python

# Crear una matriz con las listas
datos = np.array([float_salud, float_investigacion, float_militar, float_educacion])

# Calcular la matriz de covarianza
covarianza_matriz = np.cov(datos)

print("Matriz de Covarianza:")
print(covarianza_matriz)

"""

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                              INCISO A.2)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#


# Calcular las desviaciones estándar de cada variable
desviaciones_estandar = np.sqrt(np.diag(Matriz_Covarianza))

# Calcular la matriz de correlación
matriz_correlacion = Matriz_Covarianza / np.outer(desviaciones_estandar, desviaciones_estandar)

# Imprimir la matriz de correlación
print("Matriz de Correlación:")
print(matriz_correlacion)


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                         INCISO B.1) y B.2)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

# Calcular autovalores y autovectores
autovalores, autovectores = np.linalg.eig(Matriz_Covarianza)

print(" ")
print("Autovalores:")
print(autovalores)
print("\nAutovectores:")
print(autovectores)


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                              INCISO C)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

# Matriz original

original = np.array(Matriz_Covarianza)
#print("original",type(original))


# Obtener las dimensiones usando la propiedad shape
tamaño = original.shape

# Imprimir las dimensiones
print("tamaño matriz original:",tamaño)

"""

# Redimensionar original para que tenga forma (4, 16)
original = original.reshape(tamaño[0], tamaño[2])

# Obtener las dimensiones usando la propiedad shape
tamaño = original.shape

# Imprimir las dimensiones
print("tamaño matriz original:",tamaño)

"""

# Construir la matriz de transformación P con los autovectores como columnas
P = autovectores
#print("P",P)

# Obtener las dimensiones usando la propiedad shape
tamaño_P = P.shape

# Imprimir las dimensiones
print("tamaño P:",tamaño_P)

# Calcular la matriz inversa de P
P_inv = np.linalg.inv(P)
#print("P inversa",P_inv)

# Obtener las dimensiones usando la propiedad shape
tamaño_P_inv = P_inv.shape

# Imprimir las dimensiones
print("tamaño P inversa:",tamaño_P_inv)

#print("P inv:",type(P_inv))

# Transformar A a la base de autovectores y autovalores
original_transformado = np.dot(P_inv, original)
original_transformado = np.dot(original_transformado, P)

print("\nMatriz original en la base de autovectores y autovalores:")
print(original_transformado)


#----------------------------------------------------------------------------------------------------#