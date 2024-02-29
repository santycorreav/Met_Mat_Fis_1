"""
Programa para calcular la centroide de un triángulo (Problema 3, Sección 1.1.6)
Autores: 
Santiago Correa Vergara - 2182212
Samuel Rosado - 2181873
Juan Camacho - 2180800
"""

#Importamos matplotlib
import matplotlib.pyplot as plt 

#Vertices de nuestro triángulo (listas como vector)
vertice_A = [-7,21]
vertice_B = [6,32]
vertice_C = [2,3]

#Hallamos la coordenadas en X y en Y del centroide
coordenada_x_centroide = (vertice_A[0]+vertice_B[0]+vertice_C[0])*(1/3)
coordenada_y_centroide = (vertice_A[1]+vertice_B[1] +vertice_C[1])*(1/3)

#Obtenemos el nuevo vector centroide como lista
centroide = [coordenada_x_centroide,coordenada_y_centroide]

#Le ponemos un titulo
texto1="Centroide de un triángulo\nAutores:\nSantiago Correa Vergara-2182212"
texto2="\nSamuel Rosado - 2181873\nJuan Camacho - 2180800"
texto=texto1+texto2
plt.title(texto, fontsize = 9)

#Dibujamos los ejes
plt.xlabel("EJE X")
plt.ylabel("EJE Y")

#Dibujamos la rejilla
plt.grid(True)

#Definimos los limites
#ax = plt.gca()
#ax.set_xlim([-2, 5]) #limites en X
#ax.set_ylim([-2, 6]) #limites en Y

#Dibujamos el punto del centroide
plt.plot(centroide[0],centroide[1],"bo")

#Dibujamos los vertices como puntos
plt.plot(vertice_A[0],vertice_A[1],"go")
plt.plot(vertice_B[0],vertice_B[1],"ro")
plt.plot(vertice_C[0],vertice_C[1],"ko")

#Dibujamos las lineas que unen el triángulo (de vertice a vertice)
plt.plot([vertice_A[0],vertice_B[0]],[vertice_A[1],vertice_B[1]],"m")
plt.plot([vertice_B[0],vertice_C[0]],[vertice_B[1],vertice_C[1]],"m")
plt.plot([vertice_C[0],vertice_A[0]],[vertice_C[1],vertice_A[1]],"m")

#Ponemos el nombre de los vertices en la figura
plt.text(vertice_A[0], vertice_A[1], 'Vertice A', fontsize = 10)
plt.text(vertice_B[0], vertice_B[1], 'Vertice B', fontsize = 10)
plt.text(vertice_C[0], vertice_C[1], 'Vertice C', fontsize = 10)

#Ponemos el nombre de la centroide en la figura
plt.text(centroide[0], centroide[1], 'Centroide', fontsize = 10)

#Mostramos
plt.show()




