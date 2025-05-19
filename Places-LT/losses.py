import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   #转成概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

def GHMC(input_values, bins = 10):
    """Computes the GHM-C"""
    edges = torch.arange(bins + 1).float() / bins    
    p = torch.exp(-input_values)  #转成概率
    # 计算梯度模长    
    g = torch.abs(p.detach() - 1)
    weights = torch.zeros_like(g) 
    # n 用来统计有效的区间数。
    # 假如某个区间没有落入任何梯度模长，密度为0，需要额外考虑，不然取个倒数就无穷了。
    n = 0  # n valid bins
    # 通过循环计算落入10个bins的梯度模长数量
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            # 重点，所谓的梯度密度就是1/num_in_bin
            weights[inds] = 1 / num_in_bin 
            n += 1
    if n > 0:
        weights = weights / n
    return weights

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=False, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 /np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)                             #one-hot
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))  #取得对应位置的m   self.m_list
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)                                       #x的index位置换成x_m
        
        return F.cross_entropy(self.s*output, target, weight=self.weight)  #weight=self.weight
   
class NoiLoss(nn.Module):
    
    def __init__(self, cls_num_list, epsilon = 1, weight=None, train_cls=False, gamma=0.):
        super(NoiLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        self.epsilon = epsilon
        self.weight = weight
        self.train_cls = train_cls
        self.gamma = gamma
        
    def forward(self, out, target):
        index = torch.zeros_like(out[0], dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        output = out[0] - self.epsilon*out[1].abs()*self.m_list  
        if self.train_cls:
            return focal_loss(F.cross_entropy(output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(output, target, weight=self.weight)    

class NoiScrLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0.):
        super(NoiScrLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
        
    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
             
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))  
        
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list         
        output = torch.where(index, cosine-self.m, cosine)                    
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight)     
 
class NoiAngLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, epsilon = 0.1, weight=None, s=30, 
                 easy_margin=False, train_cls=False, gamma=0.):
        super(NoiAngLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list

        assert s > 0
        self.m = m
        self.epsilon = epsilon
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.easy_margin = easy_margin
        self.train_cls = train_cls
        self.gamma = gamma
        
    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device)
        noise = math.pi/2*self.epsilon/self.m_list.max()*self.m_list*noise 
        
        noise_m = noise.abs()+self.m
        m = torch.where(index, noise_m, noise.abs())   
        
        cos_m = torch.cos(m)
        sin_m = torch.sin(m)
        
        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            output = torch.where(cosine > 0, phi, cosine)               #theta<pi/2的地方，用theta+m替换; theta>pi/2用theta
        else:
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        # when theta not in [0,pi], use cosface instead      
            output = torch.where(cosine > th, phi, cosine - mm)         #phi中theta<pi-m的地方，是theta-m，theta > pi-m用cosface替换
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight) 

'''    
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
'''

'''
class ArcMLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, cls_num_list, s=30.0, max_m=0.50, weight=None, easy_margin=False):
        super(ArcMLoss, self).__init__()       
        m_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = 1/m_list**(1/4)   #
        m_list = m_list * (max_m /m_list.max())
        #m_list = torch.cuda.FloatTensor([0.5])
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(m_list)
        self.sin_m = torch.sin(m_list)
        self.th = torch.cos(math.pi - m_list)
        self.mm = torch.sin(math.pi - m_list) * m_list        
        
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)                         #phi中theta<pi/2的地方，是theta+m, >pi/2用theta
        else:
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        # when theta not in [0,pi], use cosface instead      
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)         #phi中theta<pi-m的地方，是theta-m，> pi-m用cosface替换
        # --------------------------- convert label to one-hot ---------------------------
        #one_hot = torch.zeros(cosine.size(), device='cuda')
        #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = torch.where(index, phi, cosine)
        # print(output)

        return F.cross_entropy(self.s*output, label, weight=self.weight)
'''