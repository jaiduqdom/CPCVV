# Training CP-CVV (k=5 slots) inside one s slot of Cross Validation for One-Shot-Learning
# Gestionamos lanzamientos considerando la memoria disponible del sistema
# Jaime Duque

getDisponible () {
	# Verificamos que alguna GPU tenga al menos 20GB de memoria libre
	query=`nvidia-smi --query-gpu=memory.free --format=csv|tail -2|awk '{print $1}'|while read -r line; do if [ $line -gt 20000 ]; then echo 1; fi; done`
	if [ -z "$query" ]; then return 0; fi
	return 1;
	}

espera () {
	# Esperamos a que haya memoria para lanzar el siguiente entrenamiento
	disponible=0
	while [ "$disponible" -eq "0" ]
	do
		sleep 5m
		getDisponible
		disponible="$?"
	done
	}

DATASET_ORIGEN=/home/user/GroceryStoreDataset-master/dataset
DATASET=$DATASET_ORIGEN/$1

# Entrenamientos de ConvNext Large
espera
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_CN_0 -b convnext_large -e 50 > $DATASET/salida_training_CN_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_CN_1 -b convnext_large -e 50 > $DATASET/salida_training_CN_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_CN_2 -b convnext_large -e 50 > $DATASET/salida_training_CN_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_CN_3 -b convnext_large -e 50 > $DATASET/salida_training_CN_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_CN_4 -b convnext_large -e 50 > $DATASET/salida_training_CN_4.log &
espera
# Entrenamientos de ResNeXt-101
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_RN_0 -b resnext101_32x8d -e 80 > $DATASET/salida_training_RN_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_RN_1 -b resnext101_32x8d -e 80 > $DATASET/salida_training_RN_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_RN_2 -b resnext101_32x8d -e 80 > $DATASET/salida_training_RN_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_RN_3 -b resnext101_32x8d -e 80 > $DATASET/salida_training_RN_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_RN_4 -b resnext101_32x8d -e 80 > $DATASET/salida_training_RN_4.log &
espera
# Entrenamientos de WideResNet-101
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_WRN_0 -b wide_resnet101_2 -e 80 > $DATASET/salida_training_WRN_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_WRN_1 -b wide_resnet101_2 -e 80 > $DATASET/salida_training_WRN_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_WRN_2 -b wide_resnet101_2 -e 80 > $DATASET/salida_training_WRN_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_WRN_3 -b wide_resnet101_2 -e 80 > $DATASET/salida_training_WRN_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_WRN_4 -b wide_resnet101_2 -e 80 > $DATASET/salida_training_WRN_4.log &
espera
# Entrenamientos de ViT-L-32
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_V32_0 -b vit_l_32 -e 80 > $DATASET/salida_training_V32_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_V32_1 -b vit_l_32 -e 80 > $DATASET/salida_training_V32_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_V32_2 -b vit_l_32 -e 80 > $DATASET/salida_training_V32_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_V32_3 -b vit_l_32 -e 80 > $DATASET/salida_training_V32_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_V32_4 -b vit_l_32 -e 80 > $DATASET/salida_training_V32_4.log &
espera
# Entrenamientos de RegNet X32
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_RGN_0 -b regnet_x_32gf -e 80 > $DATASET/salida_training_RGN_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_RGN_1 -b regnet_x_32gf -e 80 > $DATASET/salida_training_RGN_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_RGN_2 -b regnet_x_32gf -e 80 > $DATASET/salida_training_RGN_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_RGN_3 -b regnet_x_32gf -e 80 > $DATASET/salida_training_RGN_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_RGN_4 -b regnet_x_32gf -e 80 > $DATASET/salida_training_RGN_4.log &
espera
# Entrenamientos de EfficientNet B7
nohup python3 step_1_training.py --train_path $DATASET/train_0 --val_path $DATASET/val_0 -o $DATASET/checkpoint_EN_0 -b efficientnet_b7 -e 50 > $DATASET/salida_training_EN_0.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_1 --val_path $DATASET/val_1 -o $DATASET/checkpoint_EN_1 -b efficientnet_b7 -e 50 > $DATASET/salida_training_EN_1.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_2 --val_path $DATASET/val_2 -o $DATASET/checkpoint_EN_2 -b efficientnet_b7 -e 50 > $DATASET/salida_training_EN_2.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_3 --val_path $DATASET/val_3 -o $DATASET/checkpoint_EN_3 -b efficientnet_b7 -e 50 > $DATASET/salida_training_EN_3.log &
espera
nohup python3 step_1_training.py --train_path $DATASET/train_4 --val_path $DATASET/val_4 -o $DATASET/checkpoint_EN_4 -b efficientnet_b7 -e 50 > $DATASET/salida_training_EN_4.log &

