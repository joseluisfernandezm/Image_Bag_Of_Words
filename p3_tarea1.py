# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW/BOF
# Tarea 1: modelo BOW/BOF

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14


# librerias y paquetes por defecto
from p3_tests import test_p3_tarea1
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance


def construir_vocabulario(list_img_desc, vocab_size=5, max_iter=300):
    """   
    # Esta funcion utiliza K-Means para agrupar los descriptores en "vocab_size" clusters.
    #
    # Argumentos de entrada:
    # - list_array_desc: Lista 1xN con los descriptores de cada imagen. Cada posicion de la lista 
    #                   contiene (MxD) numpy arrays que representan UNO O VARIOS DESCRIPTORES 
    #                   extraidos sobre la imagen
    #                   - M es el numero de vectores de caracteristicas/features de cada imagen 
    #                   - D el numero de dimensiones del vector de caracteristicas/feature.    
    #   - vocab_size: int, numero de palabras para el vocabulario a construir.    
    #   - max_iter: int, numero maximo de iteraciones del algoritmo KMeans
    #
    # Argumentos de salida:
    #   - vocabulario: Numpy array de tamaño [vocab_size, D], 
    #                   que contiene los centros de los clusters obtenidos por K-Means
    #
    #
    # NOTA: se sugiere utilizar la funcion sklearn.cluster.KMeans
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html     
    """
    vocabulario = np.empty(shape=[vocab_size,list_img_desc[0].shape[1]]) # iniciamos la variable de salida (numpy array)

    list_img_desc_concatenate=np.concatenate(list_img_desc)#concatenamos la lista de descriptores que nos pasan de argumento

    kmeans= KMeans(n_clusters=vocab_size, n_init=10, max_iter=max_iter, random_state=0).fit(list_img_desc_concatenate)# aplicamos keans para el aprendizaje del vocabulario visual
    vocabulario=kmeans.cluster_centers_#nos quedamos con el elemento más representatico de cada cluster, que van a ser los centroides, estos definen nuestro vocabulario

    return vocabulario

def obtener_bags_of_words(list_img_desc, vocab):
    """
    # Esta funcion obtiene el Histograma Bag of Words para cada imagen
    #
    # Argumentos de entrada:
    # - list_img_desc: Lista 1xN con los descriptores de cada imagen. Cada posicion de la lista 
    #                   contiene (MxD) numpy arrays que representan UNO O VARIOS DESCRIPTORES 
    #                   extraidos sobre la imagen
    #                   - M es el numero de vectores de caracteristicas/features de cada imagen 
    #                   - D el numero de dimensiones del vector de caracteristicas/feature.  
    #   - vocab: Numpy array de tamaño [vocab_size, D], 
    #                  que contiene los centros de los clusters obtenidos por K-Means.   
    #
    # Argumentos de salida: 
    #   - list_img_bow: Array de Numpy [N x vocab_size], donde cada posicion contiene 
    #                   el histograma bag-of-words construido para cada imagen.
    #
    """
    # iniciamos la variable de salida (numpy array)

  
    # list_img_bow = np.empty(shape=[len(list_img_desc),len(vocab)])
    list_img_bow=[]

    numero_imagenes=len(list_img_desc)#numero de imagenes disponibles
    list_img_desc_concatenate=np.concatenate(list_img_desc)#concatenamos de nuevo
    matriz_coste=distance.cdist(vocab,list_img_desc_concatenate,'euclidean')#calculamos la distancia euclidea con cada centroide del vocabulario que creamos antes

    #Extraemos el numero de descriptores y de palabras de nuestro vocabulario
    (num_descript,aux)=list_img_desc_concatenate.shape
    (num_palabras,aux)=vocab.shape

    num_des_ima=int(num_descript/numero_imagenes)#vemos cuantos descriprores por imagen hay, redondeamos a entero

    #Con un bucle vamos creando el histograma BOW

    num = 0
    for j in range (0, numero_imagenes):#recorremos todas las imagenes
        num = j*num_des_ima
        hist=np.zeros((1,num_palabras))#inicializacion de histograma 
        for i in range (0, num_des_ima):
            minimo = np.min(matriz_coste[:,i+num])#me quedo con la distacina minima de la imagen actual a cada centroide
            pos = np.where(matriz_coste[:,i+num]== minimo)#buscamos con where la posicion en la matriz de costes donde este el minimo
            hist[:,pos[0][0]]= hist[:,pos[0][0]]+1 #acumulamos valores en la "palabra correspondente"
        hist = hist/num_des_ima
        list_img_bow.append(hist)#vamos añadiendo a la lista bow

    list_img_bow=np.concatenate(list_img_bow)#concatenamos la lista para retornarla en numpy
    
    return list_img_bow

if __name__ == "__main__":    
    dataset_path = './dataset/scenes15/'
    print("Practica 3 - Tarea 1 - Test autoevaluación\n")                    
    print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores ni mostrar datos
    #print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar datos