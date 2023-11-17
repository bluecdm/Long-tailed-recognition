"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from copy import deepcopy


def MLP(*channels):
    num_layers = len(channels) - 2
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(channels[-2], channels[-1]))
    return nn.Sequential(*layers)
    
def flatten(t):
    return t.reshape(t.shape[0], -1)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, mlp=False,
                 feat_dim=2048, feat_t=None, normalize=False, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.n_cls = num_classes
        self.distill = False

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=num_classes, use_norm=normalize, return_features=True)
        self.encoder_k = base_encoder(num_classes=num_classes, use_norm=normalize, return_features=True)

        if feat_t is None:
            feat_t = feat_dim
        
        if mlp:
            self.embed_q = MLP(feat_dim, feat_dim, dim)
            self.embed_k = MLP(feat_t, feat_t, dim)
        else:
            self.embed_q = nn.Linear(feat_dim, dim)
            self.embed_k = nn.Linear(feat_t, dim)
        
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.embed_q.parameters(), self.embed_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue_k", F.normalize(torch.randn(K, feat_t), dim=1))
        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))
        self.register_buffer("queue_i", torch.arange(-K, 0))
        
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        
    def train(self, mode=True):
        super().train(mode)
        if self.distill:
            self.encoder_k.eval()
            
    def set_distiller(self, encoder_k):
        self.distill = True
        self.encoder_k = encoder_k
        
        for p in self.embed_k.parameters():
            p.requires_grad = True
        for p in self.encoder_k.parameters():
            p.requires_grad = False
            
    def set_cls_weight(self, cls_weight, base_K=5):
        device = cls_weight.device
        cls_position = torch.arange(1, self.n_cls + 1, device=device) * base_K \
            + (self.K - self.n_cls * base_K) * cls_weight.cumsum(0)
        end_idx = cls_position.ceil().long()
        start_idx = end_idx.clone()
        start_idx[1:] = start_idx[:-1].clone()
        start_idx[0] = 0
        self.base_k = base_K
        self.cls_start_idx = start_idx
        self.K_per_cls = end_idx - start_idx
        for c, (s, e) in enumerate(zip(start_idx, end_idx)):
            self.queue_l[s:e] = c
        self.buffer_freq_correction = cls_weight / self.K_per_cls * self.K


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.embed_q.parameters(), self.embed_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue_wo_base(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)


        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # for simplicity
        assert self.queue_k.shape[1] == keys.shape[1]


        # replace the keys at ptr (dequeue and enqueue)
        self.queue_k[ptr:ptr + batch_size] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_w_base(self, keys, labels, in_idx):
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        in_idx = concat_all_gather(in_idx)

        intra_cls_idx = F.one_hot(labels, self.n_cls).cumsum(0) - 1
        intra_cls_idx = intra_cls_idx.gather(1, labels.unsqueeze(1)).squeeze(1)
        max_K_per_idx = self.K_per_cls.gather(0, labels)
        mask = intra_cls_idx < max_K_per_idx
        
        keys = keys[mask]
        labels = labels[mask]
        in_idx = in_idx[mask]
        intra_cls_idx = intra_cls_idx[mask]
        max_K_per_idx = max_K_per_idx[mask]
        
        cls_start_idx = self.cls_start_idx.gather(0, labels)
        offset = (self.queue_ptr.gather(0, labels) + intra_cls_idx) % max_K_per_idx
        
        target_pos = cls_start_idx + offset
        self.queue_k.scatter_(0, target_pos.unsqueeze(1).repeat(1, keys.shape[1]), keys.detach())
        self.queue_l.scatter_(0, target_pos, labels)
        self.queue_i.scatter_(0, target_pos, in_idx)
        
        samples_per_cls = labels.bincount(minlength=self.n_cls)
        self.queue_ptr = (self.queue_ptr + samples_per_cls) % self.K_per_cls
        
    def _dequeue_and_enqueue(self, keys, labels, in_idx):
        if hasattr(self, 'base_k'):
            self._dequeue_and_enqueue_w_base(keys, labels, in_idx)
        else:
            self._dequeue_and_enqueue_wo_base(keys, labels)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]


        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, im_cls, im_q, im_k, labels, in_idx):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        logit_q, feat_q = self.encoder_q(torch.cat([im_cls, im_q]))
        bs = im_q.shape[0]
        logit_q = logit_q[:bs]
        feat_q = feat_q[bs:]
        embed_q = F.normalize(self.embed_q(feat_q), dim=1)
        
        if not self.distill:
            self._momentum_update_key_encoder()  # update the key encoder
            im_k, labels, idx_unshuffle = self._batch_shuffle_ddp(im_k, labels)
            
        logit_k, feat_k = self.encoder_k(torch.cat([im_cls, im_k]))
        logit_k = logit_k[:bs]
        feat_k = feat_k[bs:]
        embed_k = F.normalize(self.embed_k(feat_k), dim=1)
        
        if not self.distill:
            embed_k, labels = self._batch_unshuffle_ddp(embed_k, labels, idx_unshuffle)

        self._dequeue_and_enqueue(feat_k, labels, in_idx)
        
        embed_b = F.normalize(self.embed_k(self.queue_k.clone().detach()), dim=1)
        
        query = embed_q
        key = embed_b
        k_labels = self.queue_l.clone().detach()
        k_idx = self.queue_i.clone().detach()
        
        
        return query, key, k_labels, k_idx, logit_q, logit_k

    def _inference(self, image):
        logit_q, _ = self.encoder_q(image)

        return logit_q

    def forward(self, im_cls, im_q=None, im_k=None, labels=None, index=None):
        if self.training:
            return self._train(im_cls, im_q, im_k, labels, index)
        else:
            return self._inference(im_cls)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor.detach()

    return output