"""
Programa para suma de vectores, ángulo entre vectores, magnitud de vectores, 
proyección de vectores, verificar si son o no coplanares, producto punto, 
(Problema 6, Sección 1.2.7)
Autores: 
Santiago Correa Vergara - 2182212
Manuel Felipe Barrera-2191324
"""

#Importamos matplotlib
import matplotlib.pyplot as plt 

#Importamos numpy
import numpy as np

#Importamos la libreria math
import math

#Adjuntamos los vectores a, b, c y d
vector_a = [1,2,3]
vector_b = [4,5,6]
vector_c = [3,2,1]
vector_d = [6,5,4]
print("vector a es:",vector_a)
print("vector b es:",vector_b)
print("vector c es:",vector_c)
print("vector d es:",vector_d)


#Los vectores a, b, c y d como arrays
vector_a_array = np.array(vector_a)
vector_b_array = np.array(vector_b)
vector_c_array = np.array(vector_c)
vector_d_array = np.array(vector_d)

e1 = np.array([1,0,0])
e2 = np.array([0,1,0])
e3 = np.array([0,0,1])

#Inciso a) 

print("Inciso a)")
op1 = vector_a_array+vector_b_array+vector_c_array+vector_d_array
print("operacion 1:",op1)
op2 = vector_a_array+vector_b_array-vector_c_array-vector_d_array
print("operacion 2:",op2)
op3 = vector_a_array-vector_b_array+vector_c_array-vector_d_array
print("operacion 3:",op3)
op4 = -vector_a_array+vector_b_array-vector_c_array+vector_d_array
print("operacion 4:",op4)

#Inciso b)

print("Inciso b)")

#Inciso c)

print("Inciso c)")
mag1 = 0
mag2 = 0
mag3 = 0
mag4 = 0


for x,y,z,w in :
	mag1 = math.sqrt(mag1 + x**2)
	mag2 = math.sqrt(mag2 + y**2)
	mag3 = math.sqrt(mag3 + z**2)
	mag4 = math.sqrt(mag4 + w**2)
print("||a||=",mag1,"||b||=",mag2,"||c||=",mag3,"||d||=",mag4)


#Inciso d)

#las entradas y salidas están en 
argumento1 = (sum(vector_a_array*vector_b_array))/(mag1*mag2)
print("argumento radian",argumento1)
argumento1 = argumento1*180/math.pi
print("argumento grados",argumento1)
#angulo1 = math.acos((productopunto1)/(mag1*mag2)) 
#print("el ángulo entre a y b es:",angulo1,"")
