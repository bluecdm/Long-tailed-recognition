import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

    
class GMLLoss(nn.Module):
    def __init__(self, beta, gamma, temperature=1.0, num_classes=1000, alpha=0.0):
        super().__init__()
        T_base = 1 / 30
        self.register_parameter(
            'log_T_c__no_wd__',
            nn.Parameter(torch.full([1], temperature).log()),
        )
        # self.register_parameter(
        #     'log_T_s__no_wd__',
        #     nn.Parameter(torch.full([1], T_base).log()),
        # )
        self.register_buffer(
            'log_T_s__no_wd__',
            torch.full([1], T_base).log(),
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_kd = DistillKL(T=4)
            
        
    @property
    def T_c(self):
        return self.log_T_c__no_wd__.exp()
    
    @property
    def T_s(self):
        return self.log_T_s__no_wd__.exp()
    
    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(self.num_classes)
        cls_weight = cls_num_list / cls_num_list.sum()
        self.cls_weight = cls_weight.to(self.log_T_s__no_wd__.device)

    def get_logS_x_y(self, logits, weights, target):
        
        max_logit = logits.amax(1, keepdim=True).detach()
        logits = logits - max_logit
        
        one_hot = F.one_hot(target, self.num_classes).float()
        sumexp_logits_per_cls = (logits.exp() * weights) @ one_hot
        sum_weights_per_cls = weights @ one_hot
        meanexp_logits_per_cls = sumexp_logits_per_cls / sum_weights_per_cls
        
        logmeanexp_logits_per_cls = meanexp_logits_per_cls.log() + max_logit
        
        return logmeanexp_logits_per_cls
        
    def forward(self, query, q_labels, q_idx, key, k_labels, k_idx, sup_logits, t_logits=None):
        device = (torch.device('cuda')
                  if query.is_cuda
                  else torch.device('cpu'))
        
        q_dot_k = query @ key.T / self.T_c.detach()
        qk_mask = torch.ones_like(q_idx[:, None] != k_idx[None, :]).float()
        logS_x_y = self.get_logS_x_y(q_dot_k, qk_mask, k_labels)
        
        q_dot_k_T = (query @ key.T).detach() / self.T_c
        qk_mask_T = (q_idx[:, None] != k_idx[None, :]).float()
        logS_x_y_T = self.get_logS_x_y(q_dot_k_T, qk_mask_T, k_labels)
        
        loss_sup = F.cross_entropy(sup_logits / self.T_s
                               + self.cls_weight.log(), q_labels)
        loss_con = self.criterion(logS_x_y
                               + self.cls_weight.log(), q_labels)
        loss_con_T = self.criterion(logS_x_y_T
                               + self.cls_weight.log(), q_labels)
        if t_logits is None:
            loss_div = 0
        else:
            loss_div = self.criterion_kd(sup_logits / self.T_s, t_logits / self.T_s)
        
        
        loss = self.gamma * loss_sup
        if self.beta > 0:
            loss += self.beta * (loss_con + loss_con_T)
        if self.alpha > 0:
            loss += self.alpha * loss_div
        
        return loss