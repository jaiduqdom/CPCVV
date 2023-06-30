"""
Evaluation of CP-CVV
Created on Junio 2022

@author: Jaime Duque
"""
import os
import nvidia_smi

# Elegimos la GPU con más memoria disponible para lanzar el proceso. Esto sirve únicamente para seleccionar una GPU de entre todas las disponibles.
# Para utilizar nvidia_smi, instalar:
# pip install nvidia-ml-py3

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
# from torchvision import transforms

from siamese import SiameseNetwork
from libs.dataset import Dataset

PATH_DATASET = '/home/user/GroceryStoreDataset-master/dataset'
PATH_SRV = PATH_DATASET + '/siamesas_CV#'

PATH_MODELO = []
PATH_MODELO.append('checkpoint_RN_#')      # Modelo ResNeXt-101 (tipo 0)
PATH_MODELO.append('checkpoint_WRN_#')     # Modelo Wide-ResNet-101 (tipo 1)
PATH_MODELO.append('checkpoint_V32_#')     # Modelo ViT-L-32 (tipo 2)
PATH_MODELO.append('checkpoint_EN_#')      # Modelo EfficientNet-B7 (tipo 3)
PATH_MODELO.append('checkpoint_RGN_#')     # Modelo Regnet_x_32gf (tipo 4)
PATH_MODELO.append('checkpoint_CN_#')      # Modelo Convnext_large (tipo 5)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def obtenerDataset(k):
    
    path_test = os.path.join(PATH_SRV.replace('#', str(k)), 'test')

    test_dataset      = Dataset(path_test, shuffle_pairs=False, augment=False)
    test_dataloader   = DataLoader(test_dataset, batch_size=1)    
    
    img1s = []
    img2s = []
    ys = []
    class1s = []
    class2s = []    
    
    for i, ((img1, img2), y, (class1, class2)) in enumerate(test_dataloader):
        img1s.append(img1)
        img2s.append(img2)
        ys.append(y)
        class1s.append(class1)
        class2s.append(class2)
    
    return img1s, img2s, ys, class1s, class2s

def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def calcularAcc(prob, cont, estimadores):
    correct_soft = 0
    correct_hard = 0
    total = 0
    
    y_true = []
    y_pred_h = []
    y_pred_s = []    
        
    prob = np.array(prob) / float(estimadores)
    for i in range(len(img1s)):       
        y = ys[i]
        # Estos campos nos valen para calcular el error
        y_true.append(y.cpu().detach().numpy()[0][0])
        y_pred_h.append(float(cont[i]) / float(estimadores))
        y_pred_s.append(prob[i].cpu().detach().numpy()[0])
        
        # Ahora actualizamos los valores para calcular tanto el soft como el hard voting        
        if cont[i] > int(estimadores/2):
            cont[i] = 1.0
        else:
            cont[i] = 0.0
        if prob[i] > 0.5:
            prob[i] = 1.0
        else:
            prob[i] = 0.0
            
        if cont[i] == y.data[0]:
            correct_hard += 1
        if prob[i] == y.data[0]:
            correct_soft += 1

    total = len(img1s)

    loss_soft = BinaryCrossEntropy(np.array(y_pred_s).reshape(-1, 1), np.array(y_true, dtype=float).reshape(-1, 1))[0]
    loss_hard = BinaryCrossEntropy(np.array(y_pred_h).reshape(-1, 1), np.array(y_true, dtype=float).reshape(-1, 1))[0]
    acc_hard = float(correct_hard) / float(total)
    acc_soft = float(correct_soft) / float(total)

    return acc_hard, acc_soft, loss_hard, loss_soft


# Eliminar de código final a publicar
def evaluarModeloIndividual(tipo, estimadores, img1s, img2s, ys, class1s, class2s, k):
    
    # Para mostrar mayor distancia con el estimador global, cogemos el que menor resultado ofrece
    peor_loss_soft = 0.0
    peor_loss_hard = 0.0
    peor_soft_acc = 9999999.99
    peor_hard_acc = 9999999.99
    
    loss_soft = 0.0
    loss_hard = 0.0
    acc_soft = 0.0
    acc_hard = 0.0

    prob = [0.0] * len(img1s)
    cont = [0.0] * len(img1s)
    
    # Aquí mostramos el número máximo de estimadores
    for i in range(5):
        
        if estimadores == 1:
            prob = [0.0] * len(img1s)    
            cont = [0.0] * len(img1s)
        
        checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[tipo].replace('#', str(i)), 'best.pth'))

        model = SiameseNetwork(backbone=checkpoint['backbone'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            for j in range(len(img1s)):
                # print(str(j))
                img1 = img1s[j]
                img2 = img2s[j]
                y = ys[j]
                class1 = class1s[j]
                class2 = class2s[j]
                
                #print("[{} / {}]".format(i, len(val_dataloader)))
        
                img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
                class1 = class1[0]
                class2 = class2[0]

                # Como son k estimadores, sumamos la probabilidad y la dividimos entre el número de estimadores (método SOFT)
                res = model(img1, img2)
                # Este código nos sirve para hacer hard-voting
                if res > 0.5:
                    cont[j] +=1
                prob[j] = prob[j] + res.cpu().data[0]
                
                del res
                del img1
                del img2
                del y
                torch.cuda.empty_cache()
        model.to('cpu')
        del model
        del checkpoint
        torch.cuda.empty_cache() 
        
        if estimadores == 1:
            acc_hard, acc_soft, loss_hard, loss_soft = calcularAcc(prob, cont, estimadores)
   
            if acc_hard < peor_hard_acc:
                peor_hard_acc = acc_hard
                peor_loss_soft = loss_soft
                peor_loss_hard = loss_hard
                peor_soft_acc = acc_soft

    if estimadores == 1:
        acc_hard = peor_hard_acc
        loss_soft = peor_loss_soft
        loss_hard = peor_loss_hard
        acc_soft = peor_soft_acc
    else:
        acc_hard, acc_soft, loss_hard, loss_soft = calcularAcc(prob, cont, estimadores)
            
    print("Test Soft-CVV: Loss={:.2f}\t Accuracy={:.2f}\t".format(loss_soft, acc_soft))
    print("Test Hard-CVV: Loss={:.2f}\t Accuracy={:.2f}\t".format(loss_hard, acc_hard))
    print("----------------------------------------------------------------------------")
    return loss_soft, loss_hard, acc_soft, acc_hard
    
def evaluarModelo(tipo, estimadores, img1s, img2s, ys, class1s, class2s, k):
    print("############################################################################")
    if tipo < 6:
        #PATH_OUTPUT = os.path.join(PATH_SRV, "test_" + PATH_MODELO[tipo].replace('#', str(estimadores)) + "_estimadores")
        print("Evaluación Modelo " + (PATH_MODELO[tipo].replace('out_','')).replace('_#', '') +  " con " + str(estimadores) + " estimadores...")
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
        
    # os.makedirs(PATH_OUTPUT, exist_ok=True)
    if tipo < 6:
        loss_soft, loss_hard, soft_acc, hard_acc = evaluarModeloIndividual(tipo, estimadores, img1s, img2s, ys, class1s, class2s, k)
        return loss_soft, loss_hard, soft_acc, hard_acc

    correct_soft = 0
    correct_hard = 0
    total = 0
    
    prob = [0.0] * len(img1s)    
    cont = [0.0] * len(img1s)
    
    for i in range(estimadores):
        # Si el tipo es menor que 5, entonces es un estimador individual
        checkpoint = None
        if tipo < 6:
            checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[tipo].replace('#', str(i)), 'best.pth'))
        elif tipo == 6:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[0].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[1].replace('#', str(i - 5)), 'best.pth'))
            elif i >=10 and i < 15:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[2].replace('#', str(i - 10)), 'best.pth'))
            elif i >=15 and i < 20:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[3].replace('#', str(i - 15)), 'best.pth'))
            elif i >=20 and i < 25:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[4].replace('#', str(i - 20)), 'best.pth'))
            elif i >=25 and i < 30:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[5].replace('#', str(i - 25)), 'best.pth'))
        elif tipo == 7:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[3].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[5].replace('#', str(i - 5)), 'best.pth'))
        elif tipo == 8:
            if i < 5:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[3].replace('#', str(i)), 'best.pth'))
            elif i >=5 and i < 10:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[4].replace('#', str(i - 5)), 'best.pth'))
            elif i >=10 and i < 15:
                checkpoint = torch.load(os.path.join(PATH_SRV.replace('#', str(k)), PATH_MODELO[5].replace('#', str(i - 10)), 'best.pth'))

        model = SiameseNetwork(backbone=checkpoint['backbone'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            for j in range(len(img1s)):
                # print(str(j))
                img1 = img1s[j]
                img2 = img2s[j]
                y = ys[j]
                class1 = class1s[j]
                class2 = class2s[j]
                
                #print("[{} / {}]".format(i, len(val_dataloader)))
        
                img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
                class1 = class1[0]
                class2 = class2[0]

                # Como son k estimadores, sumamos la probabilidad y la dividimos entre el número de estimadores (método SOFT)
                res = model(img1, img2)
                # Este código nos sirve para hacer hard-voting
                if res > 0.5:
                    cont[j] +=1
                prob[j] = prob[j] + res.cpu().data[0]
                
                del res
                del img1
                del img2
                del y
                torch.cuda.empty_cache()
        model.to('cpu')
        del model
        del checkpoint
        torch.cuda.empty_cache() 
        
    y_true = []
    y_pred_h = []
    y_pred_s = []    
        
    prob = np.array(prob) / float(estimadores)
    for i in range(len(img1s)):       
        y = ys[i]
        # Estos campos nos valen para calcular el error
        y_true.append(y.cpu().detach().numpy()[0][0])
        y_pred_h.append(float(cont[i]) / float(estimadores))
        y_pred_s.append(prob[i].cpu().detach().numpy()[0])
        
        # Ahora actualizamos los valores para calcular tanto el soft como el hard voting        
        if cont[i] > int(estimadores/2):
            cont[i] = 1.0
        else:
            cont[i] = 0.0
        if prob[i] > 0.5:
            prob[i] = 1.0
        else:
            prob[i] = 0.0
            
        if cont[i] == y.data[0]:
            correct_hard += 1
        if prob[i] == y.data[0]:
            correct_soft += 1

    total = len(img1s)

    loss_soft = BinaryCrossEntropy(np.array(y_pred_s).reshape(-1, 1), np.array(y_true, dtype=float).reshape(-1, 1))[0]
    loss_hard = BinaryCrossEntropy(np.array(y_pred_h).reshape(-1, 1), np.array(y_true, dtype=float).reshape(-1, 1))[0]
   
    print("Test Soft-CVV: Loss={:.2f}\t Accuracy={:.2f}\t".format(loss_soft, correct_soft / total))
    print("Test Hard-CVV: Loss={:.2f}\t Accuracy={:.2f}\t".format(loss_hard, correct_hard / total))
    print("----------------------------------------------------------------------------")
    return loss_soft, loss_hard, float(correct_soft) / float(total), float(correct_hard) / float(total)

if __name__ == "__main__":
    # Realizamos una evaluación global de los modelos.
    # Evaluamos cada modelo de manera individual, mediante CVV, y la combinación de todos mediante CVV
    # Finally, we carried a cross-validation with s = 4 (as in the paper). This was to distinguish clearly 
    # between CP-CVV slots (5) and CVV slots (4). In the code somewhere like this it says k but it refers to s.
    
    K_FOLDS = 4
    acc_soft_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    acc_hard_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    los_soft_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    los_hard_k = np.zeros((2*len(PATH_MODELO) + 3, K_FOLDS))
    
    for k in range(K_FOLDS):
        img1s, img2s, ys, class1s, class2s = obtenerDataset(k)
        for e in range(len(PATH_MODELO)):
            l_s, l_h, a_s, a_h = evaluarModelo(e, 1, img1s, img2s, ys, class1s, class2s, k)
            acc_soft_k[2*e][k] += a_s
            acc_hard_k[2*e][k] += a_h
            los_soft_k[2*e][k] += l_s
            los_hard_k[2*e][k] += l_h
            
            l_s, l_h, a_s, a_h = evaluarModelo(e, 5, img1s, img2s, ys, class1s, class2s, k)
            acc_soft_k[2*e+1][k] += a_s
            acc_hard_k[2*e+1][k] += a_h
            los_soft_k[2*e+1][k] += l_s
            los_hard_k[2*e+1][k] += l_h
        
        l_s, l_h, a_s, a_h = evaluarModelo(6, 30, img1s, img2s, ys, class1s, class2s, k)
        acc_soft_k[12][k] += a_s
        acc_hard_k[12][k] += a_h
        los_soft_k[12][k] += l_s
        los_hard_k[12][k] += l_h

        l_s, l_h, a_s, a_h = evaluarModelo(7, 10, img1s, img2s, ys, class1s, class2s, k)
        acc_soft_k[13][k] += a_s
        acc_hard_k[13][k] += a_h
        los_soft_k[13][k] += l_s
        los_hard_k[13][k] += l_h        

        l_s, l_h, a_s, a_h = evaluarModelo(8, 15, img1s, img2s, ys, class1s, class2s, k)
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
        print(MODELOS[m] + "; " + str(a_s) + "; " + str(a_h) + "; " + str(l_s) + "; " + str(l_h))
