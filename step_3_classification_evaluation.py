import os
import nvidia_smi
import sys

# CP-CVV
# This problem transforms the Siamese network into a regular classification problem. An evaluation of 
# each image is performed with the rest of the pairs in order to see if it correctly catalogs the image.

# We choose the GPU with the most available memory to launch the process. This is only to select a GPU
# from all available GPUs.
# To use nvidia_smi, install:
# pip install nvidia-ml-py3
# Author: Jaime Duque

print("Seleccionando el dispositivo con mayor memoria libre disponible...")
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()

gpu_seleccionada = 0
maximo_libre = 0.0

for i in range(deviceCount):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    
    libre = float(info.free)
    if libre > maximo_libre:
        maximo_libre = libre
        gpu_seleccionada = i
    
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

nvidia_smi.nvmlShutdown()
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_seleccionada)

print("Dispositivo seleccionado: " + str(gpu_seleccionada))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
from siamese import SiameseNetwork as SiameseNetwork
import glob
from PIL import Image
import time
from operator import truediv
from sklearn.metrics import balanced_accuracy_score

PATH_DATASET = '/home/user/GroceryStoreDataset-master/dataset'
PATH_SRV = PATH_DATASET + '/siamesas_CV0'
PATH_TEST = os.path.join(PATH_SRV, 'test')

PATH_MODELO = []
PATH_MODELO.append('checkpoint_RN_#')       # Modelo ResNeXt-101 (tipo 0)
PATH_MODELO.append('checkpoint_WRN_#')      # Modelo Wide-ResNet-101 (tipo 1)
PATH_MODELO.append('checkpoint_V32_#')      # Modelo ViT-L-32 (tipo 2)
PATH_MODELO.append('checkpoint_EN_#')       # Modelo EfficientNet-B7 (tipo 3)
PATH_MODELO.append('checkpoint_RGN_#')      # Modelo Regnet_x_32gf (tipo 4)
PATH_MODELO.append('checkpoint_CN_#')       # Modelo Convnext_large (tipo 5)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paso 1. Leemos todas las imágenes de test en memoria aplicando las transformaciones correspondientes
feed_shape = [3, 224, 224]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(feed_shape[1:])
])

image_paths = glob.glob(os.path.join(PATH_TEST, "*/*.jpg"))
image_classes = []
image_memoria = []

for image_path in image_paths:
    image_class = image_path.split(os.path.sep)[-2]
    image_classes.append(image_class)

    img = Image.open(image_path).convert("RGB")
    
    img = transform(img).float()
    img = img.unsqueeze(0)    
    image_memoria.append(img.cuda())    
      

image_classes = np.array(image_classes)
clases_diferentes, cuenta = np.unique(image_classes, return_counts=True)
numero_clases = len(clases_diferentes)

# Paso 2. Para cada imagen, seleccionamos aleatoriamente una clase de cada categoría
def generarComprobaciones():
    random.seed(int(time.time()))
    
    # En comprobaciones almacenamos los índices a comprobar para cada imagen
    comprobaciones = []
    for i in range(len(image_paths)):
        # image_path = image_paths[i]
        listaClases = []        
        for clase in clases_diferentes:
            posibles = np.where(image_classes == clase)[0]
            seleccion = i
            while (seleccion == i):
                seleccion = random.choice(posibles)
            listaClases.append(seleccion)
        comprobaciones.append(listaClases)
    return comprobaciones

#sys.exit()

# Paso 3. Evaluamos una imagen contra las imágenes representativas de cada clase y devolvemos las probabilidades de cada caso
def evaluar(indiceImg, listaImgs, modelo):    
    
    resultado = []
    with torch.no_grad():    
        for idx in listaImgs:
            res = modelo(image_memoria[indiceImg], image_memoria[idx])
            sal = res.cpu().detach().numpy()[0]
            resultado.append(sal[0])
    return resultado
        
# Paso 4. Evaluación global de un modelo. Realiza la evaluación de todas las imágenes seleccionada    
def evaluarGlobal(modelo, comprobaciones ):
    resultado = []
    for i in range(len(image_paths)):
        res = evaluar(i, comprobaciones[i], modelo)
        resultado.append(res)
    return np.array(resultado, dtype = 'float')

# Paso 5. Calcular la matriz de confusión a partir de los resultados obtenidos durante la predicción del modelo.
def buscarClase(clase):
    for j in range(len(clases_diferentes)):    
        if clases_diferentes[j] == clase:
            return j
    return len(clases_diferentes)

# Calculamos tanto y_true, y_pred como el accuracy tanto top-1 como top-2
def calcularPred(resultados):
    # Añadimos una clase ya que es posible que una imagen no se clasifique en ninguna de las categorías, por ejemplo si todas las
    # siamesas devuelven que las dos imágenes no pertenecen a la misma categoría.
    y_true = []
    y_pred = []
    aciertos = 0.0
    aciertos_top_2 = 0.0    # Si alguna de las 2 clases ganadoras es la correcta
    # Para cada imagen analizamos la clase que debería tener y en cuál le ha asignado
    for i in range(len(image_classes)):
        real = image_classes[i]
        # La clase ganadora será la que mayor probabilidad tenga 
        maximo = 0.0
        ganadora = ""
        maximo_2 = 0.0
        ganadora_2 = ""
        for j in range(len(clases_diferentes)):
            # Nos quedamos con la máxima, independientemente de si la siamesa la acepta como ganadora o no. Esto lo hacemos
            # ya que consideramos el sistema un clasificador entre las clases de test y no contemplamos el caso de que no
            # se detecte nada.
   
            if resultados[i][j] > maximo:
                maximo_2 = maximo
                ganadora_2 = ganadora
                maximo = resultados[i][j]
                ganadora = clases_diferentes[j]
            elif resultados[i][j] > maximo_2:
                maximo_2 = resultados[i][j]
                ganadora_2 = clases_diferentes[j]
        y_true.append(buscarClase(real))
        y_pred.append(buscarClase(ganadora))
        if real == ganadora:
            aciertos += 1.0
        if real == ganadora_2 or real == ganadora:
            aciertos_top_2 += 1.0
    accuracy = float(aciertos) / float(len(image_classes))
    accuracy_top_2 = float(aciertos_top_2) / float(len(image_classes))    
    return y_true, y_pred, accuracy, accuracy_top_2

# Calculamos la matriz de confusión y otros datos
def mostrarDatos(num_clases, y_pred, y_true):
    MC = np.zeros((num_clases,num_clases))

    # Listas para calcular el balanced accuracy    
    # y_true = []
    # y_pred = []

    aciertos = 0
    totales = 0
    for i in range(len(y_pred)):
        totales += 1
        if (y_pred[i] == y_true[i]):
            aciertos += 1
        
        MC[int(y_true[i])][int(y_pred[i])] += 1

    print("Número de imágenes acertadas: " + str(aciertos))
    print("Número de imágenes en total: " + str(totales))    
    print("Test accuracy: " + str(round(float(aciertos/totales),3)))

    tp = np.diag(MC)

    a = np.sum(MC, axis=0)
    for i in range(len(a)):
        if a[i] == 0.0:
            a[i] = 0.0000000000001
            
    prec = list(map(truediv, tp, a))
    
    b = np.sum(MC, axis=1)
    for i in range(len(b)):
        if b[i] == 0.0:
            b[i] = 0.0000000000001    
        
    rec = list(map(truediv, tp, b))
    
    prec2 = prec.copy()
    rec2 = rec.copy()
    for i in range(len(prec2)):
        prec2[i] = round(prec2[i],3)
    for i in range(len(rec2)):
        rec2[i] = round(rec2[i],3)
    
    print ('Precision: {}\nRecall: {}'.format(prec2, rec2))
    print ('Precision (media): {}\nRecall (media): {}'.format(round(np.mean(prec),3), round(np.mean(rec),3)))
    # print("Matriz Confusión: \n" + str(MC))
    
    # Balanced accuracy
    bac = balanced_accuracy_score(y_true, y_pred)
    print("Balanced Accuracy: " + str(bac))    
    
    p = np.mean(prec)
    r = np.mean(rec)
    f1 = 2.0 * p * r / (p + r)
    print ('F1-Score: {}\n'.format(round(f1,3)))

    nc = len(MC)
    
    for i in range(nc):
        suma = 0
        for j in range(nc):
            suma += MC[i][j]
        for j in range(nc):
            MC[i][j] = float(MC[i][j]) / float(suma)
    
    
    RELACION_CLASES = clases_diferentes.copy()
    # RELACION_CLASES.append("No Class")
    
    print('\\newcommand\\items{"' + str(nc) + '"}')
    print('\\arrayrulecolor{white}\\noindent')
    print('\\begin{tabular}{cc*{\\items}{|E}|}')
    print('\\multicolumn{1}{c}{} &\\multicolumn{1}{c}{} &\\multicolumn{\\items}{c}{Predicted} \\\\ \\hhline{~*\\items{|-}|}')
    print('\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & ')
    for i in range(nc-1):
        print('\\multicolumn{1}{c}{\\rot{' + RELACION_CLASES[i] + '}} & ')
    print('\\multicolumn{1}{c}{\\rot{' + RELACION_CLASES[nc-1] + '}} \\\\ \\hhline{~*\\items{|-}|}')
    print('\\multirow{\\items}{*}{\\rotatebox{90}{Actual}} ')
    
    for i in range(nc):
        cad = '&' + RELACION_CLASES[i]
        for j in range(nc):
            cad = cad + ' &' + str(round(MC[i][j], 2))
        cad = cad + ' \\\\ \\hhline{~*\\items{|-}|}'
        print(cad)
    print('\\end{tabular}')
    

def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

#def cross_entropy(predictions, targets, epsilon=1e-12):
def CrossEntropy(targets, predictions, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

def evaluarModelo(tipo, estimadores, comprobaciones):
    print("############################################################################")
    if tipo < 6:
        #PATH_OUTPUT = os.path.join(PATH_SRV, "test_" + PATH_MODELO[tipo].replace('#', str(estimadores)) + "_estimadores")
        print("Evaluación Modelo " + (PATH_MODELO[tipo].replace('checkpoint_','')).replace('_#', '') +  " con " + str(estimadores) + " estimadores...")
    elif tipo == 6:
        #PATH_OUTPUT = os.path.join(PATH_SRV, 'test_out_CVV_Completo')
        print("Evaluación Modelo CVV Global...")
        # El tipo 7 se queda con los 2 estimadores que han obtenido mejores resultados: Convnext_large y EfficientNet-B7
    elif tipo == 7:
        #PATH_OUTPUT = os.path.join(PATH_SRV, 'test_out_CVV_Selectivo_Convnext_large_y_EfficientNet_B7')
        print("Evaluación Modelo CVV Selectivo...(Convnext_large_y_EfficientNet_B7)")
    elif tipo == 8:
        #PATH_OUTPUT = os.path.join(PATH_SRV, 'test_out_CVV_Selectivo_Convnext_large_y_EfficientNet_B7_Regnet_x_32gf')
        print("Evaluación Modelo CVV Selectivo...(Convnext_large_y_EfficientNet_B7_Regnet_x_32gf)")
    print("----------------------------------------------------------------------------")

    resultados = None
    resultados_hard = None
    inicio = 0
    inicio_hard = 0    
    
    for i in range(estimadores):
        # Si el tipo es menor que 5, entonces es un estimador individual
        checkpoint = None
        if tipo < 6:
            checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[tipo].replace('#', str(i)), 'best.pth'))
        elif tipo == 6:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[0].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[1].replace('#', str(i - 5)), 'best.pth'))
            elif i >=10 and i < 15:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[2].replace('#', str(i - 10)), 'best.pth'))
            elif i >=15 and i < 20:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[3].replace('#', str(i - 15)), 'best.pth'))
            elif i >=20 and i < 25:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[4].replace('#', str(i - 20)), 'best.pth'))
            elif i >=25 and i < 30:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[5].replace('#', str(i - 25)), 'best.pth'))
        elif tipo == 7:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[3].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[5].replace('#', str(i - 5)), 'best.pth'))
        elif tipo == 8:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[3].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[4].replace('#', str(i - 5)), 'best.pth'))
            elif i >=10 and i < 15:
                checkpoint = torch.load(os.path.join(PATH_SRV, PATH_MODELO[5].replace('#', str(i - 10)), 'best.pth'))

        model = SiameseNetwork(backbone=checkpoint['backbone'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        resultados_e = evaluarGlobal(model, comprobaciones)
        if inicio == 0:
            inicio = 1
            resultados = resultados_e.copy()
        else:
            resultados = resultados + resultados_e

        for v in range(len(resultados_e)):
            for w in range(len(resultados_e[v])):
                if resultados_e[v][w] > 0.5:
                    resultados_e[v][w] = 1.0
                else:
                    resultados_e[v][w] = 0.0
        
        if inicio_hard == 0:
            inicio_hard = 1
            resultados_hard = resultados_e
        else:
            resultados_hard = resultados_hard + resultados_e

    resultados = np.array(resultados, dtype='float') / float(estimadores)
    
    print(str(resultados))
    y_true, y_pred, accuracy, accuracy_top_2 = calcularPred(resultados)    
    mostrarDatos(len(clases_diferentes), y_pred, y_true)
    loss_soft = CrossEntropy(np.array(y_true), np.array(y_pred))
    
    resultados_hard = np.array(resultados_hard, dtype='float') / float(estimadores)
    print(str(resultados_hard))
    y_true_hard, y_pred_hard, accuracy_hard, accuracy_top_2_hard = calcularPred(resultados_hard)
    loss_hard = CrossEntropy(np.array(y_true_hard), np.array(y_pred_hard))
 
    print("Test Soft-CVV: Loss={:.4f}\t Accuracy={:.4f}\t TOP-2 Accuracy={:.4f}\t".format(loss_soft, accuracy, accuracy_top_2))
    print("Test Hard-CVV: Loss={:.4f}\t Accuracy={:.4f}\t TOP-2 Accuracy={:.4f}\t".format(loss_hard, accuracy_hard, accuracy_top_2_hard))
    print("----------------------------------------------------------------------------")
    return loss_soft, loss_hard, accuracy, accuracy_hard, 

if __name__ == "__main__":
    K_FOLDS = 5
    # Realizamos una evaluación global de los modelos.
    # Evaluamos cada modelo de manera individual, mediante CVV, y la combinación de todos mediante CVV
    acc_soft_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    acc_hard_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    los_soft_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    los_hard_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    
    for k in range(K_FOLDS):
        c = generarComprobaciones()
        for e in range(len(PATH_MODELO)):
            l_s, l_h, a_s, a_h = evaluarModelo(e, 1, c)
            acc_soft_k[2*e][k] += a_s
            acc_hard_k[2*e][k] += a_h
            los_soft_k[2*e][k] += l_s
            los_hard_k[2*e][k] += l_h
            
            l_s, l_h, a_s, a_h = evaluarModelo(e, 5, c)
            acc_soft_k[2*e+1][k] += a_s
            acc_hard_k[2*e+1][k] += a_h
            los_soft_k[2*e+1][k] += l_s
            los_hard_k[2*e+1][k] += l_h
        
        l_s, l_h, a_s, a_h = evaluarModelo(6, 30, c)
        acc_soft_k[12][k] += a_s
        acc_hard_k[12][k] += a_h
        los_soft_k[12][k] += l_s
        los_hard_k[12][k] += l_h

        l_s, l_h, a_s, a_h = evaluarModelo(7, 10, c)
        acc_soft_k[13][k] += a_s
        acc_hard_k[13][k] += a_h
        los_soft_k[13][k] += l_s
        los_hard_k[13][k] += l_h        

        l_s, l_h, a_s, a_h = evaluarModelo(8, 15, c)
        acc_soft_k[14][k] += a_s
        acc_hard_k[14][k] += a_h
        los_soft_k[14][k] += l_s
        los_hard_k[14][k] += l_h
        
    print("Acc Soft K:")
    print(str(acc_soft_k))
    print("Acc Hard K:")
    print(str(acc_hard_k))
    print("Loss Soft K:")
    print(str(los_soft_k))
    print("Loss Hard K:")
    print(str(los_hard_k))
    
    MODELOS = ['ResNeXt-101',       'ResNeXt-101 (CVV con 5 estimadores)', 
               'Wide-ResNet-101',   'Wide-ResNet-101 (CVV con 5 estimadores)', 
               'ViT-L-32',          'ViT-L-32 (CVV con 5 estimadores)', 
               'EfficientNet-B7',   'EfficientNet-B7 (CVV con 5 estimadores)', 
               'Regnet_x_32gf',     'Regnet_x_32gf (CVV con 5 estimadores)', 
               'Convnext_large',    'Convnext_large (CVV con 5 estimadores)', 
               'CVV Global (5 estimadores por modelo)', 
               'CVV Selectivo: Convnext_large + EfficientNet-B7  (5 estimadores por modelo)',
               'CVV Selectivo: Convnext_large + Regnet_x_32gf + EfficientNet-B7  (5 estimadores por modelo)']

    print("Modelo; Accuracy (Soft); Accuracy (Hard); Loss (Soft); Loss (Hard)")
    for m in range(len(acc_soft_k)):
        a_s = acc_soft_k[m].sum() / float(K_FOLDS)
        a_h = acc_hard_k[m].sum() / float(K_FOLDS)
        l_s = los_soft_k[m].sum() / float(K_FOLDS)
        l_h = los_hard_k[m].sum() / float(K_FOLDS)
        print(MODELOS[m] + "; " + str(np.round(a_s, 4) ) + "; " + str(np.round(a_h, 4)) + "; " + str(np.round(l_s, 4)) + "; " + str(np.round(l_h, 4)))
