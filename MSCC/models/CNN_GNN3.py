from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset
from .embed import DataEmbedding, PositionalEmbedding

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from itertools import product
from torch_geometric.nn import GCNConv, GATConv  # 用于GNN部分

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

def transfer_entropy(series_x, series_y, lag=1, bins="auto"):
    """
    计算从X到Y的传递熵
    series_x: 源时间序列
    series_y: 目标时间序列
    lag: 滞后步长
    bins: 离散化的分箱数
    """
    if bins == "auto":
        bins = int(np.log2(len(series_x))) + 1
    # 离散化时间序列
    def discretize(series, bins):
        min_val, max_val = np.min(series), np.max(series)
        edges = np.linspace(min_val - 1e-6, max_val + 1e-6, bins + 1)
        return np.digitize(series, edges)
    
    x = discretize(series_x, bins)
    y = discretize(series_y, bins)
    
    y_t = y[lag:]      # y(t)
    y_tm = y[:-lag]    # y(t-1)
    x_tm = x[:-lag]    # x(t-1)
    
    # 计算联合概率分布
    def joint_prob(*args):
        levels = [np.unique(arg) for arg in args]
        joint = np.zeros(tuple(len(l) for l in levels))
        
        for idxs in product(*[range(len(l)) for l in levels]):
            mask = np.ones_like(args[0], dtype=bool)
            for i, arg in enumerate(args):
                mask &= (arg == levels[i][idxs[i]])
            joint[idxs] = np.sum(mask)
        
        joint /= np.sum(joint)
        return joint, levels
    
    # 计算条件熵 H(Yt|Yt-1)
    joint_yy, _ = joint_prob(y_t, y_tm)
    p_yy = joint_yy.sum(axis=0)  # p(y_tm)
    cond_entropy_yy = -np.sum(joint_yy * np.log2(joint_yy / (p_yy + 1e-10) + 1e-10))
    
    # 计算条件熵 H(Yt|Yt-1, Xt-1)
    joint_yxy, _ = joint_prob(y_t, y_tm, x_tm)
    p_yx = joint_yxy.sum(axis=0)  # p(y_tm, x_tm)
    cond_entropy_yxy = -np.sum(joint_yxy * np.log2(joint_yxy / (p_yx + 1e-10) + 1e-10))
    
    # 传递熵 TE = H(Yt|Yt-1) - H(Yt|Yt-1, Xt-1)
    te = cond_entropy_yy - cond_entropy_yxy
    return max(te, 0)  # 传递熵应为非负

def get_causal_graph(X, lag=1):
    """
    计算 n 个变量的因果图
    X: 输入数据，形状为 (num_time_steps, num_variables)
    lag: 滞后步长
    返回：因果图矩阵
    """
    num_variables = X.shape[1]
    cause_graph = np.zeros((num_variables, num_variables))  # 初始化因果图矩阵
    
    # 计算每一对变量的传递熵
    for i in range(num_variables):
        for j in range(num_variables):
            if i != j:
                # 计算从 X[i] 到 X[j] 的传递熵
                cause_graph[i, j] = transfer_entropy(X[:, i], X[:, j], lag)
    
    return cause_graph

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

# ========== 3. CNN 提取单变量特征 ==========
class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 n_features,
                 num_channel=[32, 32, 40],
                 kernel_size=4,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.2,
                 hidden_activation='relu',
                 device='cpu',
                 causal_graph=None,
                 scale1 = 64,
                 scale2 = 128,
                 scale3 = 256,
                 init_alpha = 1.0
                 ):

        # initialize the super class
        super(CNNFeatureExtractor, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.num_channel = num_channel
        self.device = device
        self.activation = get_activation_by_name(hidden_activation)
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha = init_alpha
        self.conv_layers = nn.Sequential()
        # self.causal_graph = nn.Parameter(torch.tensor(causal_graph,dtype=torch.float32))
        self.causal_graph = nn.Parameter(causal_graph.clone().detach().requires_grad_(True).float())
        prev_channels = self.n_features
        self.alpha1 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha2 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha3 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha4 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        for idx, out_channels in enumerate(self.num_channel[:-1]):
            self.conv_layers.add_module("conv" + str(idx),torch.nn.Conv1d(prev_channels, self.num_channel[idx + 1], self.kernel_size, self.stride))
            self.conv_layers.add_module(self.hidden_activation + str(idx),self.activation)
            self.conv_layers.add_module("pool" + str(idx), nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.fc = nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
            torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.num_channel[-1], self.n_features)
        )
        self.pooling_layer = nn.AdaptiveMaxPool1d(64)
    
    def forward(self, x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L] 
        # x = x.contiguous().view(b, c, l)
        scales = [self.scale1, self.scale2, self.scale3, l]  # 直接用 l 代表 scale4
        x_scales = [x[:, :, -s:] for s in scales]   
        for i in range(1, 4):  # 对 scale2, scale3, scale4 进行池化
            x_scales[i] = self.pooling_layer(x_scales[i])
        x_scales = [self.conv_layers(x_s) for x_s in x_scales]
        x_scales = [self.fc(x_s) for x_s in x_scales]
        outputs = torch.zeros(1, b, self.n_features).to(self.alpha1.device)
        x_scales_unsq = []
        output_channel_independent = torch.zeros(b, self.n_features).to(self.alpha1.device)
        output_channel_dependent = torch.zeros(b, self.n_features).to(self.alpha1.device)  
        for i, alpha in enumerate([self.alpha1, self.alpha2, self.alpha3, self.alpha4]):
            output_channel_independent += alpha * x_scales[i]
            x_scales_unsq.append(x_scales[i].unsqueeze(1))#
        output_channel_dependent = 0*torch.matmul(output_channel_independent, self.causal_graph.T)+output_channel_independent 
        outputs = torch.unsqueeze(output_channel_dependent, dim=0)
        x = sum(alpha * x_s for alpha, x_s in zip([self.alpha1, self.alpha2, self.alpha3, self.alpha4], x_scales_unsq))
        return x,[],outputs,self.causal_graph

class TemEnc(nn.Module):
    def __init__(self, c_in, d_model, win_size, seq_size, tr,cnn):
        super(TemEnc, self).__init__()
        self.enc = cnn
        self.pro = nn.Sequential(
            nn.Linear(c_in, c_in),
            nn.GELU(),
            nn.Linear(c_in, c_in),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        ex = x
        filters = torch.ones(1,1,self.seq_size).to(device)
        ex2 = ex ** 2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr[:,:,self.seq_size-1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr2[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr2[:,:,self.seq_size-1:] /= self.seq_size
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / ltrm.sum(-1)
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]
        ux, att,_,_= self.enc(unmasked_tokens)
        rec = self.pro(ux)
        att.append(rec)
        _,_,output,causal_graph = self.enc(x)
        return att,output,causal_graph

class FreEnc(nn.Module):
    def __init__(self, c_in, d_model, win_size, fr,cnn):
        super(FreEnc, self).__init__()
        self.enc = cnn
        self.pro = nn.Sequential(
            nn.Linear(c_in, c_in),
            nn.GELU(),
            nn.Linear(c_in,c_in),
            nn.Sigmoid()
        )
        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))
        self.fr = fr
    
    def forward(self, x):
        # x: [B, T, C]
        # ex = self.emb(x) # [B, T, D]
        ex = x # [B, T, D]
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]
        ix = torch.fft.irfft(cx).transpose(1,2)
        dx, att,_,_ = self.enc(ix)
        rec = self.pro(dx)
        att.append(rec)
        return att

class Contrative_CNN(nn.Module):
    def __init__(self, win_size, n_features,seq_size = 10,d_model=100,fr=0.1, tr=0.1, dev=None,kernel_size=4,stride = 1,dropout_rate=0.4,hidden_activation='relu',num_channel=[32, 32, 40],scale1 = 64,scale2 = 128,scale3 = 256,init_alpha = 1.0,causal_graph=None):
        super(Contrative_CNN, self).__init__()
        global device
        device = dev
        self.win_size = win_size
        self.seq_size = seq_size
        self.d_model = d_model
        self.fr = fr
        self.tr = tr
        self.c_in = n_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.num_channel = num_channel
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha = init_alpha
        self.causal_graph = causal_graph
        print("self.fr:",self.fr)
        print("self.tr:",self.tr)
        self.cnn = CNNFeatureExtractor(n_features=self.c_in, num_channel=self.num_channel, causal_graph=self.causal_graph,device=dev,scale1=self.scale1,scale2=self.scale2,scale3 =self.scale3,init_alpha = self.init_alpha)
        self.tem = TemEnc(c_in = self.c_in, d_model=self.d_model, win_size=self.win_size, seq_size=self.seq_size, tr = self.tr,cnn=self.cnn)
        self.fre = FreEnc(c_in=self.c_in, d_model=self.d_model, win_size=self.win_size, fr=self.fr,cnn=self.cnn)

    def forward(self, x):
        tematt,output,causal_graph = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt,output,causal_graph,freatt
    
    
class CNN():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 validation_size=0.2,
                 num_channel=[32, 32, 32],
                 kernel_size=4,stride = 1,dropout_rate=0.1,hidden_activation='relu',
                 fr=0.05,
                 tr=0.05,
                 scale1 = 20,scale2 = 100,scale3 = 200,init_alpha = 1.0,):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.window_size = window_size
        print("self.window_size:",self.window_size)
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.feats = feats
        self.num_channel = num_channel
        self.lr = lr
        self.validation_size = validation_size
        self.tr = tr
        self.fr = fr
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha = init_alpha
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        self.lamda_con =0.00000000000001
        
    def fit(self, data):
        print("调用multi_contrastive_cnn模型进行训练")
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]
        
        # if os.path.exists('causal_graph_tensor_smd.pt'):
        #     print("已存在缓存文件，正在加载传递信息熵矩阵...")
        #     initial_graph_tensor = torch.load('causal_graph_tensor_smd.pt')
        # else:
        #     causal_graph_matrix = get_causal_graph(tsTrain, lag=1)
        #     initial_graph_tensor = torch.tensor(causal_graph_matrix, dtype=torch.float32)
        #     print("传递信息熵矩阵:\n", initial_graph_tensor.shape)
        #     torch.save(initial_graph_tensor, 'causal_graph_tensor_smd.pt')
        
        #其他三个数据集
        causal_graph_matrix = get_causal_graph(tsTrain, lag=1)
        initial_graph_tensor = torch.tensor(causal_graph_matrix, dtype=torch.float32)
        print("传递信息熵矩阵:\n", initial_graph_tensor.shape)
        torch.save(initial_graph_tensor, 'causal_graph_tensor.pt')
        # print(initial_graph_tensor)
        self.causal_graph = initial_graph_tensor
        self.model = Contrative_CNN(n_features=self.feats, num_channel=self.num_channel, causal_graph=self.causal_graph,win_size = self.window_size,dev=self.device,scale1=self.scale1,scale2=self.scale2,scale3 =self.scale3,init_alpha = self.init_alpha,kernel_size=self.kernel_size,stride = self.stride,dropout_rate=self.dropout_rate,hidden_activation=self.hidden_activation,fr = self.fr,tr= self.tr).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        state_dict = self.model.state_dict()
        print("Available parameters:")
        # for key in state_dict.keys():
            # print(key)
        # for name, param in self.model.named_parameters():
            # print(name, id(param))
        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            lambda_reg = 0.1
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                tematt,output,causal_graph,freatt = self.model(x)
                threshold=0
                causal_reg_loss = 0
                for i in range(causal_graph.shape[0]):
                    for j in range(causal_graph.shape[0]):
                        if causal_graph[i, j] > threshold:
                            p_i = output[:, :,i]
                            p_j = output[:, :,j]
                            p_i_centered = p_i - torch.mean(p_i)
                            p_j_centered = p_j - torch.mean(p_j)
                            beta_ij = torch.sum(p_i_centered * p_j_centered) / (torch.sum(p_i_centered * p_i_centered) + 1e-8)
                            residual = p_j_centered - beta_ij * p_i_centered
                            residual_var = torch.sum(residual * residual) / output.shape[1]#(batch_size)
                            
                            # Add to causal regularization loss, weighted by causal strength
                            causal_reg_loss += causal_graph[i,j] * residual_var
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                        my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                                tematt[u])))    
                    con_loss += (torch.mean(my_kl_loss(
                        (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                        tematt[u].detach())) + torch.mean(
                        my_kl_loss(tematt[u].detach(), (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))
                adv_loss = adv_loss / len(freatt)   #ghl个人解释：拉远样本中不相似的部分 —  
                con_loss = con_loss / len(freatt)   #ghl个人解释：拉近样本中相似的部分 +
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)
                
                loss_rec = self.loss(output, target)+ lambda_reg * causal_reg_loss
                loss = self.lamda_con*(con_loss - adv_loss) + loss_rec
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    tematt,output,causal_graph,freatt= self.model(x)
                    adv_loss = 0.0
                    con_loss = 0.0
                    for u in range(len(freatt)):
                        adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                            my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                                    tematt[u])))
                        con_loss += (torch.mean(my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach())) + torch.mean(
                            my_kl_loss(tematt[u].detach(), (
                                    freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                    adv_loss = adv_loss / len(freatt)
                    con_loss = con_loss / len(freatt)
                    
                    threshold=0
                    causal_reg_loss = 0
                    for i in range(causal_graph.shape[0]):
                        for j in range(causal_graph.shape[0]):
                            if causal_graph[i, j] > threshold:
                                p_i = output[:, :,i]
                                p_j = output[:, :,j]
                                p_i_centered = p_i - torch.mean(p_i)
                                p_j_centered = p_j - torch.mean(p_j)
                                # Calculate regression coefficient beta_ij
                                beta_ij = torch.sum(p_i_centered * p_j_centered) / (torch.sum(p_i_centered * p_i_centered) + 1e-8)
                                
                                # Calculate residual (the part of p_j not explained by p_i)
                                residual = p_j_centered - beta_ij * p_i_centered
                                
                                # Variance of the residual (how much of p_j is not explained by p_i)
                                residual_var = torch.sum(residual * residual) / output.shape[1]
                                
                                # Add to causal regularization loss, weighted by causal strength
                                causal_reg_loss += causal_graph[i,j] * residual_var
                            
                    output = output.view(-1, self.feats*self.pred_len)
                    target = target.view(-1, self.feats*self.pred_len)
                    
                    loss_rec = self.loss(output, target)+ lambda_reg * causal_reg_loss
                    loss = self.lamda_con*(con_loss - adv_loss) + loss_rec
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                    print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        lambda_reg = 0
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                tematt,output,causal_graph,freatt= self.model(x)
                threshold=0
                causal_reg_loss = 0
                #预测阶段，暂时不采用causal_reg_loss和对比学习loss,只采用predcition loss
                # for i in range(causal_graph.shape[0]):
                #     for j in range(causal_graph.shape[0]):
                #         if causal_graph[i, j] > threshold:
                #             p_i = output[:, :,i]
                #             p_j = output[:, :,j]
                #             p_i_centered = p_i - torch.mean(p_i)
                #             p_j_centered = p_j - torch.mean(p_j)
                #             # Calculate regression coefficient beta_ij
                #             beta_ij = torch.sum(p_i_centered * p_j_centered) / (torch.sum(p_i_centered * p_i_centered) + 1e-8)
                            
                #             # Calculate residual (the part of p_j not explained by p_i)
                #             residual = p_j_centered - beta_ij * p_i_centered
                            
                #             # Variance of the residual (how much of p_j is not explained by p_i)
                #             residual_var = torch.sum(residual * residual) / output.shape[1]
                            
                #             # Add to causal regularization loss, weighted by causal strength
                #             causal_reg_loss += causal_graph[i,j] * residual_var
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)
                mse = torch.sub(output, target).pow(2)+ lambda_reg * causal_reg_loss

                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        # scores = 0.5 * (torch.log(self.sigma + self.eps) + (scores - self.mu)**2 / (self.sigma+self.eps))
        
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        y_hats = torch.cat(y_hats, dim=0)
        y_hats = y_hats.numpy()
        
        l, w = y_hats.shape
        
        # new_scores = np.zeros((l - self.pred_len, w))
        # for i in range(w):
        #     new_scores[:, i] = scores[self.pred_len - i:l-i, i]
        # scores = np.mean(new_scores, axis=1)
        # scores = np.pad(scores, (0, self.pred_len - 1), 'constant', constant_values=(0,0))
        
        # new_y_hats = np.zeros((l - self.pred_len, w))
        # for i in range(w):
        #     new_y_hats[:, i] = y_hats[self.pred_len - i:l-i, i]
        # y_hats = np.mean(new_y_hats, axis=1)
        # y_hats = np.pad(y_hats, (0, self.pred_len - 1), 'constant',constant_values=(0,0))

        assert scores.ndim == 1
        # self.y_hats = y_hats
        
        print('scores: ', scores.shape)
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[: self.window_size+self.pred_len-1] = scores[0]
            padded_decision_scores_[self.window_size+self.pred_len-1 : ] = scores

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
