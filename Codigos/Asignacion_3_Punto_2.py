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

#----------------------------------------------------------------------------------------------------#

import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
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

# Matriz de datos en float

float_datos = [[float_salud],
               [float_investigacion],
               [float_militar],
               [float_educacion]
]

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

#plt.show()

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
cov_S = sum((float_salud[i] - media_salud) * (float_salud[i] - media_salud) for i in range(len(float_salud))) / (len(float_salud))

# Calcular la covarianza entre Investigacion e Investigacion (Varianza de Salud, Var(I) )
cov_I = sum((float_investigacion[i] - media_investigacion) * (float_investigacion[i] - media_investigacion) for i in range(len(float_investigacion))) / len(float_investigacion)

# Calcular la covarianza entre Militar y Militar (Varianza de Salud, Var(M) )
cov_M = sum((float_militar[i] - media_militar) * (float_militar[i] - media_militar) for i in range(len(float_militar))) / len(float_militar)

# Calcular la covarianza entre Educacion y Educacion (Varianza de Salud, Var(E) )
cov_E = sum((float_educacion[i] - media_educacion) * (float_educacion[i] - media_educacion) for i in range(len(float_educacion))) / len(float_educacion)

#----------------------------------------------------------------------------------------------------#

#  Las covarianzas entre S/I, S/M, S/E 

# Calcular la covarianza entre Salud e Investigacion (S/I)
cov_S_I = sum((float_salud[i] - media_salud) * (float_investigacion[i] - media_investigacion) for i in range(len(float_salud))) / len(float_salud)
cov_I_S = cov_S_I

# Calcular la covarianza entre Salud y Militar (S/M)
cov_S_M = sum((float_salud[i] - media_salud) * (float_militar[i] - media_militar) for i in range(len(float_salud))) / len(float_salud)
cov_M_S = cov_S_M

# Calcular la covarianza entre Salud y Educacion (S/E)
cov_S_E = sum((float_salud[i] - media_salud) * (float_educacion[i] - media_educacion) for i in range(len(float_salud))) / len(float_salud)
cov_E_S = cov_S_E



#----------------------------------------------------------------------------------------------------#

# Las covarianzas entre M/I, E/I, E/M

# Calcular la covarianza entre Militar e Investigacion (M/I)
cov_M_I = sum((float_militar[i] - media_militar) * (float_investigacion[i] - media_investigacion) for i in range(len(float_militar))) / len(float_militar)
cov_I_M = cov_M_I

# Calcular la covarianza entre Educacion e Investigacion (E/I)
cov_E_I = sum((float_educacion[i] - media_educacion) * (float_investigacion[i] - media_investigacion) for i in range(len(float_educacion))) / len(float_educacion)
cov_I_E = cov_E_I

# Calcular la covarianza entre Educacion y Militar (E/M)
cov_E_M = sum((float_educacion[i] - media_educacion) * (float_militar[i] - media_militar) for i in range(len(float_educacion))) / len(float_educacion)
cov_M_E = cov_E_M


#----------------------------------------------------------------------------------------------------#

# Entonces la matriz de covarianza es

Matriz_Covarianza = [
                    [cov_S, cov_S_I, cov_S_M, cov_S_E],
                    [cov_I_S, cov_I, cov_I_M, cov_I_E],
                    [cov_M_S, cov_M_I, cov_M, cov_M_E],
                    [cov_E_S, cov_E_I, cov_E_M, cov_E]
                        ]


# Comparación de función covarianza creada con la integrada en Python

# Crear una matriz con las listas
datos = np.array([float_salud, float_investigacion, float_militar, float_educacion])

# Calcular la matriz de covarianza
Matriz_Covarianza_python = np.cov(datos)



#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                              INCISO A.2)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#


# Calcular las desviaciones estándar de cada variable
desviaciones_estandar = np.sqrt(np.diag(Matriz_Covarianza))

# Calcular la matriz de correlación
matriz_correlacion = Matriz_Covarianza / np.outer(desviaciones_estandar, desviaciones_estandar)


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                         INCISO B.1) y B.2)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

# Calcular autovalores y autovectores
autovalores, autovectores = np.linalg.eig(Matriz_Covarianza)

autovalores = [autovalores]

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#                                              INCISO C)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

# Matriz original

original = np.array(Matriz_Covarianza)

# Obtener las dimensiones usando la propiedad shape
tamaño = original.shape

# Construir la matriz de transformación P con los autovectores como columnas
P = autovectores

# Obtener las dimensiones usando la propiedad shape
tamaño_P = P.shape

# Calcular la matriz inversa de P
P_inv = np.linalg.inv(P)

# Obtener las dimensiones usando la propiedad shape
tamaño_P_inv = P_inv.shape


# Transformar A a la base de autovectores y autovalores
matriz_transformacion = np.dot(P_inv, original)
matriz_transformacion = np.dot(matriz_transformacion, P)


#----------------------------------------------------------------------------------------------------#

# IMPRESION DE TODO


print("-----Listas con formato 4 decimal-----")
print(" ")
Matriz_Covarianza = formato_lista(Matriz_Covarianza)
Matriz_Covarianza_python = formato_lista(Matriz_Covarianza_python)
matriz_correlacion = formato_lista(matriz_correlacion)
autovalores = formato_lista(autovalores)
autovectores = formato_lista(autovectores)
matriz_transformacion = formato_lista(matriz_transformacion)


# Usar un bucle para imprimir cada fila en una nueva línea
for k in Matriz_Covarianza:
    print(k)

print("Matriz Covarianza")
print(Matriz_Covarianza_python)
print("Matriz Correlacion")
print(matriz_correlacion)
print("Autovalores")
print(autovalores)
print("Autovectores")
print(autovectores)
print("Matriz Transformacion")
print(matriz_transformacion)
print(" ")

#----------------------------------------------------------------------------------------------------#

# LISTAS GUARDADAS COMO MATRIZ SINTAXIS LATEX EN .TXT

archivo_M_covarianza = "matriz_covarianza.txt"
archivo_M_covarianza_python = "matriz_covarianza_python.txt"
archivo_M_correlacion = "matriz_correlacion.txt"
archivo_autovalores = "autovalores.txt"
archivo_autovectores = "autovectores.txt"
archivo_M_transformacion = "matriz_transformacion.txt"

guardar_lista_latex(Matriz_Covarianza, archivo_M_covarianza)
guardar_lista_latex(Matriz_Covarianza_python, archivo_M_covarianza_python)
guardar_lista_latex(matriz_correlacion, archivo_M_correlacion)
guardar_lista_latex(autovalores, archivo_autovalores)
guardar_lista_latex(autovectores, archivo_autovectores)
guardar_lista_latex(matriz_transformacion, archivo_M_transformacion)

#----------------------------------------------------------------------------------------------------#