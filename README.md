# Long-Tailed Recognition by Mutual Information Maximization between Latent Features and Ground-Truth Labels

Python code for ICML 2023 paper entitled "Long-Tailed Recognition by Mutual Information Maximization between Latent Features and Ground-Truth Labels"

#### Test environment:
*    Ubuntu 18.04
*    CUDA 11.3, cuDNN 8.3.2


#### Requirements:
*    Python 3.8
*    Pytorch 1.12.1
*    tqdm
*    scikit-learn
*    tensorboard
*    matplotlib


#### This code is based on:
*    https://github.com/facebookresearch/moco
*    https://github.com/dvlab-research/Parametric-Contrastive-Learning/tree/main/PaCo/LT


#### Experiment procedure:
1. (Set dataset directory) Change --data argument of sh/ImageNetLT_train_teacher.sh and sh/ImageNetLT_train_student.sh
2. (Train teacher) Run bash ./sh/ImageNetLT_train_teacher.sh
3. (Train student) Run bash ./sh/ImageNetLT_train_student.sh


#### Long-tailed recognition accuracy:

Dataset | Backbone | Epochs | Top-1 Acc(%)
---- | :----: | :----: | :----: 
ImageNet-LT | ResNeXt-50 |  90 | 58.3
ImageNet-LT | ResNeXt-50 |400 | 58.8
iNaturalist 2018 | ResNet-50 | 100 | 73.1
iNaturalist 2018 | ResNet-50 | 400 | 74.5
CIFAR-100-LT (imb. 100) | ResNet-32 | 200 | 53.0
CIFAR-100-LT (imb. 50) | ResNet-32 | 200 | 57.6
CIFAR-100-LT (imb. 10) | ResNet-32 | 200 | 65.7
CIFAR-100-LT (imb. 100) | ResNet-32 | 400 | 54.0
CIFAR-100-LT (imb. 50) | ResNet-32 | 400 | 58.1
CIFAR-100-LT (imb. 10) | ResNet-32 | 400 | 67.0
