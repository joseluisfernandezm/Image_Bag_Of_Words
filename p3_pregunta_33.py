# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW
# Memoria - Pregunta 3.3

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14



from p3_utils import load_image_dataset,create_results_webpage
from p3_tarea1 import construir_vocabulario, obtener_bags_of_words
from p3_tarea2 import obtener_features_tiny, obtener_features_hog
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier


'''
En este ejercicio nos piden crear un esquema de clasificacion de imagenes con los siguientes detalles:

+ Características: HOG con parámetro tam=100
+ Modelo: BOW con max_iter=10
+ Clasificador: Random Forest
+ Otros: Ratio train_test=0.20, y Máx número datos por categoría: 200

'''

# Primero: cargamos el dataset

tam_diccionario=50# que es el optimo del ejercicio 3.2.1

dataset_path = './dataset/scenes15/'
max_num_categoria=200

datos=load_image_dataset(dataset_path,description=None,categories=None,load_content=False,shuffle=True, random_state=0,resize_shape=None,max_per_category=max_num_categoria,debug=False)

# Segundo: pasamos al entrenamiento y el test

train_test=0.20

x=datos.filenames
y=datos.target
[x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size=train_test, random_state = 42)#fijamos random_state a 42

# Tercero: HOG
tam=100 #para ejercicio base


list_img_desc_hog_train = obtener_features_hog(x_train, tamano = tam, orientaciones = 9, pixeles_por_celda = (8, 8),celdas_bloque = (4,4))
list_img_desc_hog_test = obtener_features_hog(x_test, tamano = tam, orientaciones = 9, pixeles_por_celda = (8, 8),celdas_bloque = (4,4))

list_img_desc_tiny_train=obtener_features_tiny(x_train, tamano = tam)
list_img_desc_tiny_test=obtener_features_tiny(x_test, tamano = tam)

# Cuarto: Vocabulario BOW con datos train
max_iter=10
vocabulario_train_hog =  construir_vocabulario(list_img_desc_hog_train, vocab_size = tam_diccionario, max_iter = max_iter)

vocabulario_train_tiny =  construir_vocabulario(list_img_desc_tiny_train, vocab_size = tam_diccionario, max_iter = max_iter)

# Quinto: Obtener descriptores BOW para train/test con el vocabulario de test
descriptores_BOW_train_hog = obtener_bags_of_words(list_img_desc_hog_train, vocabulario_train_hog)
descriptores_BOW_test_hog = obtener_bags_of_words(list_img_desc_hog_test, vocabulario_train_hog)

descriptores_BOW_train_tiny = obtener_bags_of_words(list_img_desc_tiny_train, vocabulario_train_tiny)
descriptores_BOW_test_tiny = obtener_bags_of_words(list_img_desc_tiny_test, vocabulario_train_tiny)

# Sexto: Usar Random Forest para entrenar

n = 100# decidimos variar n_estimators y dejamos fijo random_state a 42

forest_hog = RandomForestClassifier(n_estimators = n, random_state=42) # random_state=42

forest_hog.fit(descriptores_BOW_train_hog, y_train) # Entrenamiento

Rendimiento_test_hog = forest_hog.score(descriptores_BOW_test_hog, y_test)
Rendimiento_train_hog = forest_hog.score(descriptores_BOW_train_hog, y_train)


forest_tiny = RandomForestClassifier(n_estimators = n, random_state=42) # random_state=42

forest_tiny.fit(descriptores_BOW_train_tiny, y_train) # Entrenamiento

Rendimiento_test_tiny = forest_tiny.score(descriptores_BOW_test_tiny, y_test)
Rendimiento_train_tiny = forest_tiny.score(descriptores_BOW_train_tiny, y_train)


print('Tamaño Diccionario = ',tam_diccionario)
print('Rendimiento TEST con Hog= ',Rendimiento_test_hog)
print('Rendimiento TRAIN con Hog = ',Rendimiento_train_hog)
print('Rendimiento TEST con Tiny= ',Rendimiento_test_tiny)
print('Rendimiento TRAIN con Tiny = ',Rendimiento_train_tiny)


#--Creacion webpage--
# 7: Mat Confusion
categories = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial','InsideCity',
                'Kitchen', 'LivingRoom', 'Mountain', 'Office','OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']

abbr_categories = ['Bed','Cst','For','HWy', 'Ind','Cty','Kit','Liv','Mnt',
                    'Off','OC','Sto','St','Sub','Bld']

# Poner en formato texto las etiquetas, tener las catergorias reales
[col_y_train] = y_train.shape
y_train_words = []
for i in range (0, col_y_train):
    k = y_train[i]
    y_train_words.append(categories[k])

[col_y_test] = y_test.shape
y_test_words = []
for i in range (0, col_y_test):
    p = y_test[i]
    y_test_words.append(categories[p])

y_pred = forest_hog.predict(descriptores_BOW_test_hog)
y_pred_words = []
[col_y_pred] = y_test.shape
for i in range (0, col_y_test):
    m = y_pred[i]
    y_pred_words.append(categories[m])


confusion_matrix = create_results_webpage(x_train, x_test, y_train_words, y_test_words, categories, abbr_categories, y_pred_words, 'pregunta_33')
