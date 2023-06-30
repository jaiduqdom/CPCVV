# -*- coding: utf-8 -*-
""" CP-CVV
    We create k different validation groups and a fixed test group.

    We unify all data under the whole directory and generate s = 5 folders for cross-validation.
    Within each folder there will be another k = 5 folders to implement CVV.
    We use GroceryStoreDataset
    
    Autor: Jaime Duque
    Fecha: 25-Mayo-2022
"""
import os
import shutil

# Ubicación del dataset
PATH_DATASET = '/home/user/GroceryStoreDataset-master/dataset'

PATH_DATASET_TRAIN = os.path.join(PATH_DATASET, 'train')
PATH_DATASET_VAL = os.path.join(PATH_DATASET, 'val')
PATH_DATASET_TEST = os.path.join(PATH_DATASET, 'test')

PATH_DATASET_TRAIN_PROCESADO = os.path.join(PATH_DATASET, 'trainProcesado')
PATH_DATASET_TEST_PROCESADO = os.path.join(PATH_DATASET, 'testProcesado')


RELACION_CLASES = ['Alpro-Blueberry-Soyghurt', 'Alpro-Fresh-Soy-Milk', 'Alpro-Shelf-Soy-Milk', 'Alpro-Vanilla-Soyghurt', 
                   'Anjou', 'Arla-Ecological-Medium-Fat-Milk', 'Arla-Ecological-Sour-Cream', 'Arla-Lactose-Medium-Fat-Milk', 
                   'Arla-Medium-Fat-Milk', 'Arla-Mild-Vanilla-Yoghurt', 'Arla-Natural-Mild-Low-Fat-Yoghurt', 
                   'Arla-Natural-Yoghurt', 'Arla-Sour-Cream', 'Arla-Sour-Milk', 'Arla-Standard-Milk', 'Asparagus', 
                   'Aubergine', 'Avocado', 'Banana', 'Beef-Tomato', 'Bravo-Apple-Juice', 'Bravo-Orange-Juice', 
                   'Brown-Cap-Mushroom', 'Cabbage', 'Cantaloupe', 'Carrots', 'Conference', 'Cucumber', 'Floury-Potato', 
                   'Galia-Melon', 'Garant-Ecological-Medium-Fat-Milk', 'Garant-Ecological-Standard-Milk', 'Garlic', 'Ginger', 
                   'God-Morgon-Apple-Juice', 'God-Morgon-Orange-Juice', 'God-Morgon-Orange-Red-Grapefruit-Juice', 
                   'God-Morgon-Red-Grapefruit-Juice', 'Golden-Delicious', 'Granny-Smith', 'Green-Bell-Pepper', 
                   'Honeydew-Melon', 'Kaiser', 'Kiwi', 'Leek', 'Lemon', 'Lime', 'Mango', 'Nectarine', 'Oatly-Natural-Oatghurt', 
                   'Oatly-Oat-Milk', 'Orange', 'Orange-Bell-Pepper', 'Papaya', 'Passion-Fruit', 'Peach', 'Pineapple', 
                   'Pink-Lady', 'Plum', 'Pomegranate', 'Red-Beet', 'Red-Bell-Pepper', 'Red-Delicious', 'Red-Grapefruit', 
                   'Regular-Tomato', 'Royal-Gala', 'Satsumas', 'Solid-Potato', 'Sweet-Potato', 'Tropicana-Apple-Juice', 
                   'Tropicana-Golden-Grapefruit', 'Tropicana-Juice-Smooth', 'Tropicana-Mandarin-Morning', 'Valio-Vanilla-Yoghurt', 
                   'Vine-Tomato', 'Watermelon', 'Yellow-Bell-Pepper', 'Yellow-Onion', 'Yoggi-Strawberry-Yoghurt', 
                   'Yoggi-Vanilla-Yoghurt', 'Zucchini']

def buscarClase(nombre):
    for i in range(len(RELACION_CLASES)):
        if nombre == RELACION_CLASES[i]:
            return i
    return -1

# Definimos 5 grupos 
K = 5

CLASES_TEST = [ ['Arla-Ecological-Sour-Cream', 'Brown-Cap-Mushroom', 'Carrots', 'God-Morgon-Orange-Juice', 'God-Morgon-Red-Grapefruit-Juice', 
                 'Granny-Smith', 'Lime', 'Mango', 'Oatly-Oat-Milk', 'Passion-Fruit', 'Sweet-Potato', 'Tropicana-Golden-Grapefruit', 'Yellow-Onion'], # CV 0
                ['Alpro-Blueberry-Soyghurt', 'Arla-Mild-Vanilla-Yoghurt', 'Cabbage', 'God-Morgon-Apple-Juice', 'God-Morgon-Orange-Red-Grapefruit-Juice', 
                 'Honeydew-Melon', 'Nectarine', 'Oatly-Natural-Oatghurt', 'Red-Bell-Pepper', 'Red-Delicious', 'Satsumas', 'Tropicana-Juice-Smooth', 'Zucchini'], # CV 1
                ['Alpro-Fresh-Soy-Milk', 'Alpro-Vanilla-Soyghurt', 'Banana', 'Beef-Tomato', 'God-Morgon-Orange-Red-Grapefruit-Juice', 
                 'Kiwi', 'Orange-Bell-Pepper', 'Papaya', 'Solid-Potato', 'Tropicana-Mandarin-Morning', 'Watermelon', 'Yellow-Bell-Pepper', 'Yoggi-Strawberry-Yoghurt'], # CV 2
                ['Alpro-Blueberry-Soyghurt', 'Arla-Lactose-Medium-Fat-Milk', 'Arla-Natural-Mild-Low-Fat-Yoghurt', 'Cantaloupe', 'Carrots', 
                 'Conference', 'Garlic', 'Honeydew-Melon', 'Pomegranate', 'Tropicana-Juice-Smooth', 'Vine-Tomato', 'Yoggi-Vanilla-Yoghurt', 'Zucchini'], # CV 3
                ['Arla-Standard-Milk', 'Asparagus', 'Aubergine', 'Carrots', 'Conference', 'Floury-Potato', 'Galia-Melon', 'Garant-Ecological-Medium-Fat-Milk', 
                 'Golden-Delicious', 'Oatly-Natural-Oatghurt', 'Pineapple', 'Tropicana-Juice-Smooth', 'Yoggi-Vanilla-Yoghurt'] # CV 4
                ]

def crearDirectorio(directorio):
    # Crear directorio si no existe y limpiarlo si existe
    try:
        shutil.rmtree(directorio)
    except:
        pass    
    try: 
        os.mkdir(directorio)
    except:
        pass

def moverDataset(origen, destino):
    crearDirectorio(destino)
    
    total = 0
    
    files1 = os.listdir(origen)
    for file1 in files1:
        
        #print(file1)
        # Excluimos los ficheros
        if os.path.isfile(os.path.join(origen, file1)):
            continue

        files2 = os.listdir(os.path.join(origen, file1))
        for file2 in files2:
                     
            # Excluimos los ficheros
            if os.path.isfile(os.path.join(origen, file1, file2)):
                continue
            
            # Creamos el directorio
            # crearDirectorio(os.path.join(destino, file2))            

            files3 = os.listdir(os.path.join(origen, file1, file2))
            for file3 in files3:
                
                # Movemos las imágenes
                if os.path.isfile(os.path.join(origen, file1, file2, file3)):
                    try: 
                        os.mkdir(os.path.join(destino, file2))
                    except:
                        pass                    
                    shutil.copy(os.path.join(origen, file1, file2, file3), os.path.join(destino, file2, file3))
                    total += 1
                else:
                    files4 = os.listdir(os.path.join(origen, file1, file2, file3))
                    for file4 in files4:
                        # Movemos las imágenes
                        if os.path.isfile(os.path.join(origen, file1, file2, file3, file4)):
                            try: 
                                os.mkdir(os.path.join(destino, file3))
                            except:
                                pass                    
                            shutil.copy(os.path.join(origen, file1, file2, file3, file4), os.path.join(destino, file3, file4))
                            total += 1
    return total
                     
t = moverDataset(PATH_DATASET_TRAIN, PATH_DATASET_TRAIN_PROCESADO)
t = moverDataset(PATH_DATASET_TEST,  PATH_DATASET_TEST_PROCESADO)

def copiarDataset(origen, destino, tipo):
    files1 = os.listdir(origen)
    for file1 in files1:
        
        # Excluimos los ficheros
        if os.path.isfile(os.path.join(origen, file1)):
            continue

        # Creamos las carpetas
        try: 
            os.mkdir(os.path.join(destino, file1))
        except:
            pass

        files2 = os.listdir(os.path.join(origen, file1))
        for file2 in files2:
                     
            # Copiamos los ficheros
            if os.path.isfile(os.path.join(origen, file1, file2)):
                shutil.copy(os.path.join(origen, file1, file2), os.path.join(destino, file1, tipo + "_" + file2))

crearDirectorio(os.path.join(PATH_DATASET, 'completo'))
copiarDataset(os.path.join(PATH_DATASET, 'trainProcesado'), os.path.join(PATH_DATASET, 'completo'), 'train')
copiarDataset(os.path.join(PATH_DATASET, 'testProcesado'), os.path.join(PATH_DATASET, 'completo'), 'test')

def getClases(i):
    CLASES_T = []
    for c in RELACION_CLASES:
        if c not in CLASES_TEST[i]:
            CLASES_T.append(c)
    
    # En clases train y clases val vamos a tener el desglose de las clases por slot para utilizarlas en CVV.
    CLASES_TRAIN = []
    CLASES_VAL= []
    
    # Creamos los slots de train/validation
    for s in range(K):
        SUB_T = []
        SUB_V = []
        p = s
        while (p < len(CLASES_T)):
            SUB_V.append(CLASES_T[p])
            p = p + K
        for c in CLASES_T:
            if c not in SUB_V:
                SUB_T.append(c)
        CLASES_TRAIN.append(SUB_T)
        CLASES_VAL.append(SUB_V)
    return CLASES_TRAIN, CLASES_VAL

# Copiamos una carpeta selectivamente para crear el juego de test y los slots de entrenamiento
def copiarDatasetSelectivo(origen, destino, lista):
    crearDirectorio(destino)    
    files1 = os.listdir(origen)
    for file1 in files1:
        
        # Excluimos los ficheros
        if os.path.isfile(os.path.join(origen, file1)):
            continue
        
        if file1 in lista:
            # Creamos las carpetas
            # Crear directorio
            try: 
                os.mkdir(os.path.join(destino, file1))
            except:
                pass            
    
            files2 = os.listdir(os.path.join(origen, file1))
            for file2 in files2:
                         
                # Copiamos los ficheros
                if os.path.isfile(os.path.join(origen, file1, file2)):
                    shutil.copy(os.path.join(origen, file1, file2), os.path.join(destino, file1, file2))

for i in range(5):
    ruta = os.path.join(PATH_DATASET, 'siamesas_CV' + str(i))
    crearDirectorio(ruta)    
    copiarDatasetSelectivo(os.path.join(PATH_DATASET, 'completo'), os.path.join(ruta, 'test'), CLASES_TEST[i])

    CLASES_TRAIN, CLASES_VAL = getClases(i)
    for j in range(K):
        copiarDatasetSelectivo(os.path.join(PATH_DATASET, 'completo'), os.path.join(ruta, 'train_' + str(j)), CLASES_TRAIN[j])
        copiarDatasetSelectivo(os.path.join(PATH_DATASET, 'completo'), os.path.join(ruta, 'val_' + str(j)), CLASES_VAL[j])
