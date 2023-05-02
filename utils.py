import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib


def save_codes(path, prefix='__save_codes__'):
    assert not os.path.isdir(os.path.join(path, prefix)), \
        'save folder already exists'

    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    for (root, dirs, files) in os.walk('.'):
        if len(files) > 0:
            for file in files:
                if prefix in root:
                    continue
                if os.path.splitext(file)[-1] in ['.py', '.sh']:
                    src = os.path.join(root, file)
                    dst = os.path.join(path, prefix, src)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        if hasattr(train_data.dataset, 'labels'):
            training_labels = np.array(train_data.dataset.labels).astype(int)
        elif hasattr(train_data.dataset, 'targets'):
            training_labels = np.array(train_data.dataset.targets).astype(int)
        else:
            raise NotImplementedError

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
