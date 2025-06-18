import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset
from ..utils.embed import DataEmbedding, PositionalEmbedding

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)
    
class Multi_CNN(nn.Module):
    def __init__(self,
                 n_features,
                 num_channel=[32, 32, 40],
                 kernel_size=4,
                 stride=1,
                 dropout_rate=0.4,
                 hidden_activation='relu',
                 device='cpu',
                 scale1 = 64,
                 scale2 = 128,
                 scale3 = 256,
                 init_alpha = 1.0
                 ):
        super(Multi_CNN, self).__init__()
        
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_channel = num_channel
        self.device = device
        self.activation = get_activation_by_name(hidden_activation)
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha = init_alpha
        self.conv_layers = nn.Sequential()
        prev_channels = self.n_features
        self.alpha1 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha2 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha3 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        self.alpha4 = nn.Parameter(torch.tensor(self.init_alpha))  # 初始化为 0.25
        for idx, out_channels in enumerate(self.num_channel[:-1]):
            self.conv_layers.add_module(
                "conv" + str(idx),
                torch.nn.Conv1d(prev_channels, self.num_channel[idx + 1], 
                self.kernel_size, self.stride))
            self.conv_layers.add_module(self.hidden_activation + str(idx),
                                    self.activation)
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
        self.pooling_layer = nn.AdaptiveAvgPool1d(64)#nn.AdaptiveMaxPool1d(64)#
    def forward(self, x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        # x = x.contiguous().view(b, c, l)
        x_scale1 = x[:, :, -self.scale1:]   # 取最后64个数据
        x_scale2 = x[:, :, -self.scale2:] # 取最后128个数据
        x_scale3 = x[:, :, -self.scale3:] # 取最后256个数据
        x_scale4 = x             
        x_scale2 = self.pooling_layer(x_scale2)
        x_scale3 = self.pooling_layer(x_scale3)
        x_scale4 = self.pooling_layer(x_scale4)
        
        x_scale1 = self.conv_layers(x_scale1)     # [128, feature, 23]
        x_scale2 = self.conv_layers(x_scale2)
        x_scale3 = self.conv_layers(x_scale3)
        x_scale4 = self.conv_layers(x_scale4)
        
        x_scale1 = self.fc(x_scale1)     # [128, feature, 23]
        x_scale2 = self.fc(x_scale2)
        x_scale3 = self.fc(x_scale3)
        x_scale4 = self.fc(x_scale4)
        out_scale1 = torch.zeros(1, b, self.n_features).to(self.device)
        out_scale2 = torch.zeros(1, b, self.n_features).to(self.device)
        out_scale3 = torch.zeros(1, b, self.n_features).to(self.device)
        out_scale4 = torch.zeros(1, b, self.n_features).to(self.device)
        
        out_scale1[0] = torch.squeeze(x_scale1, dim=-2)
        out_scale2[0] = torch.squeeze(x_scale2, dim=-2)
        out_scale3[0] = torch.squeeze(x_scale3, dim=-2)
        out_scale4[0] = torch.squeeze(x_scale4, dim=-2)
        
        self.alpha1.data = self.alpha1.data.to(self.device)
        self.alpha2.data = self.alpha2.data.to(self.device)
        self.alpha3.data = self.alpha3.data.to(self.device)
        self.alpha4.data = self.alpha4.data.to(self.device)
        
        outputs = self.alpha1 * out_scale1 + self.alpha2 * out_scale2 + self.alpha3 * out_scale3 + self.alpha4 * out_scale4
        x_scale1 = x_scale1.unsqueeze(1)  # [B, 1, C]
        x_scale2 = x_scale2.unsqueeze(1)
        x_scale3 = x_scale3.unsqueeze(1)
        x_scale4 = x_scale4.unsqueeze(1)
        x = self.alpha1 * x_scale1 + self.alpha2 * x_scale2 + self.alpha3 * x_scale3 + self.alpha4 * x_scale4
        return x, [],outputs# 返回输出和空列表（为了与原来的Encoder接口一致）
    #forward优化版本
    # def forward(self, x):
    #     b, l, c = x.shape
    #     x = x.permute(0, 2, 1)  # [B, C, L]
        
    #     scales = [self.scale1, self.scale2, self.scale3, l]  # 直接用 l 代表 scale4
    #     x_scales = [x[:, :, -s:] for s in scales] 
        
    #     for i in range(1, 4):  # 对 scale2, scale3, scale4 进行池化
    #         x_scales[i] = self.pooling_layer(x_scales[i])
        
    #     x_scales = [self.conv_layers(x_s) for x_s in x_scales]
    #     x_scales = [self.fc(x_s) for x_s in x_scales]
        
    #     outputs = torch.zeros(1, b, self.n_features).to(self.alpha1.device)
    #     x_scales_unsq = []
        
    #     for i, alpha in enumerate([self.alpha1, self.alpha2, self.alpha3, self.alpha4]):
    #         x_scales_unsq.append(x_scales[i].unsqueeze(1))
    #         outputs += alpha * torch.squeeze(x_scales[i], dim=-2)
    #         print("outputs:",outputs.shape)
        
    #     x = sum(alpha * x_s for alpha, x_s in zip([self.alpha1, self.alpha2, self.alpha3, self.alpha4], x_scales_unsq))
        
    #     return x, [], outputs
    
class TemEnc(nn.Module):
    def __init__(self, c_in, d_model, e_layers, win_size, seq_size, tr,num_channel=[32, 32, 40],kernel_size=4,stride = 1,dropout_rate=0.4,hidden_activation='relu',scale1 = 64,scale2 = 128,scale3 = 256,init_alpha = 1.0):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha= init_alpha
        self.enc = Multi_CNN(
            n_features=d_model,
            num_channel=self.num_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dropout_rate=self.dropout_rate,
            hidden_activation=self.hidden_activation,
            scale1 =self.scale1,
            scale2 =self.scale2,
            scale3 =self.scale3,
            init_alpha = self.init_alpha
            
        )

        self.enc2 = Multi_CNN(
            n_features=c_in,
            num_channel=self.num_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dropout_rate=self.dropout_rate,
            hidden_activation=self.hidden_activation,
            scale1 =self.scale1,
            scale2 =self.scale2,
            scale3 =self.scale3,
            init_alpha = self.init_alpha
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        ex = self.emb(x) # [B, T, D]
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
        ux, att,_= self.enc(unmasked_tokens)
        rec = self.pro(ux)
        att.append(rec)
        _,_,output = self.enc2(x)
        return att,output 

class FreEnc(nn.Module):
    def __init__(self, c_in, d_model, e_layers, win_size, fr,num_channel=[32, 32, 40],kernel_size=4,stride = 1,dropout_rate=0.4,hidden_activation='relu',scale1 = 64,scale2 = 128,scale3 = 256,init_alpha = 1.0):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha= init_alpha
        
        self.enc = Multi_CNN(
            n_features=d_model,
            num_channel=self.num_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dropout_rate=self.dropout_rate,
            hidden_activation=self.hidden_activation,
            scale1 =self.scale1,
            scale2 =self.scale2,
            scale3 =self.scale3,
            init_alpha = self.init_alpha
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))

        self.fr = fr
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]
        ix = torch.fft.irfft(cx).transpose(1,2)
        dx, att,_ = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att
    
class MTFA(nn.Module):
    def __init__(self, win_size, c_in,seq_size = 10,d_model=100, e_layers=3, fr=0.4, tr=0.5, dev=None,kernel_size=4,stride = 1,dropout_rate=0.4,hidden_activation='relu',num_channel=[32, 32, 40],scale1 = 64,scale2 = 128,scale3 = 256,init_alpha = 1.0):
        super(MTFA, self).__init__()
        global device
        device = dev
        self.tem = TemEnc(c_in, d_model, e_layers, win_size, seq_size, tr,num_channel,kernel_size,stride,dropout_rate,hidden_activation,scale1,scale2,scale3,init_alpha)
        self.fre = FreEnc(c_in, d_model, e_layers, win_size, fr,num_channel,kernel_size,stride,dropout_rate,hidden_activation,scale1,scale2,scale3,init_alpha)

    def forward(self, x):
        tematt,output = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt,output,freatt
    
   
class MSCC():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 d_model = 100,
                 validation_size=0.2,
                 num_channel=[32, 32, 40],
                 kernel_size=4,stride = 1,dropout_rate=0.4,hidden_activation='relu',
                 e_layers=3,
                 fr=0.4,
                 tr=0.5,
                 scale1 = 64,scale2 = 128,scale3 = 256,init_alpha = 1.0,
                 lamda =1e-14
                 ):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.feats = feats
        self.d_model =d_model
        self.num_channel = num_channel
        self.lr = lr
        self.validation_size = validation_size
        
        self.tr = tr
        self.fr = fr
        self.e_layers = e_layers
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.init_alpha = init_alpha
        self.model = MTFA(win_size = self.window_size, c_in = self.feats,d_model=self.d_model, e_layers=self.e_layers, fr=self.fr, tr=self.tr,dev=self.device,num_channel=self.num_channel,kernel_size=self.kernel_size,stride = self.stride,dropout_rate=self.dropout_rate,hidden_activation=self.hidden_activation,scale1 = self.scale1,scale2 = self.scale2,scale3 = self.scale3,init_alpha = self.init_alpha).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None 
        self.sigma = None
        self.eps = 1e-10
        self.lamda = lamda
        # self.lamda = 0.00000000000001
        
    def fit(self, data):
        print("using MSCC")
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)

        train_steps = len(train_loader)
        state_dict = self.model.state_dict()
        # print("Available parameters:")
        # for key in state_dict.keys():
        #     print(key)
            
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (input, target) in loop:
                weights_flat1 = self.model.tem.enc2.conv_layers.conv0.weight
                weights_flat1 = weights_flat1.view(32, -1)  # [32, 550]    
                weights_norm = weights_flat1 / torch.norm(weights_flat1, dim=1, keepdim=True)  # L2 归一化
                sim_matrix = torch.matmul(weights_norm, weights_norm.t())  # [32, 32]
                mask = ~torch.eye(32, dtype=torch.bool, device=sim_matrix.device)
                sim_off_diag = sim_matrix[mask]  # 非对角线元素       
                sim_loss1 = torch.sum(sim_off_diag ** 2)
                
                # weights_flat2 = state_dict['tem.enc.conv_layers.conv0.weight'].view(32, -1)  # [32, 550]
                weights_flat2 = self.model.tem.enc.conv_layers.conv0.weight
                weights_flat2 = weights_flat2.view(32, -1)  # [32, 550]
                weights_norm = weights_flat2 / torch.norm(weights_flat2, dim=1, keepdim=True)  # L2 归一化
                sim_matrix = torch.matmul(weights_norm, weights_norm.t())  # [32, 32]
                mask = ~torch.eye(32, dtype=torch.bool, device=sim_matrix.device)
                sim_off_diag = sim_matrix[mask]  
                sim_loss2 = torch.sum(sim_off_diag ** 2)
                weights_flat3 = self.model.fre.enc.conv_layers.conv0.weight
                weights_flat3 = weights_flat3.view(32, -1)  # [32, 550]
                weights_norm = weights_flat3 / torch.norm(weights_flat3, dim=1, keepdim=True)  # L2 归一化
                sim_matrix = torch.matmul(weights_norm, weights_norm.t())  # [32, 32]
                mask = ~torch.eye(32, dtype=torch.bool, device=sim_matrix.device)
                sim_off_diag = sim_matrix[mask] 
                sim_loss3 = torch.sum(sim_off_diag ** 2)
                
                loss_kernel = sim_loss1 + sim_loss2 + sim_loss3
                self.optimizer.zero_grad()
                input, target = input.float().to(self.device), target.to(self.device)
                tematt,output,freatt = self.model(input)
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
                # print("adv_loss:",adv_loss)
                # print("con_loss:",con_loss)
                output = output.view(-1, self.feats*self.pred_len).to(self.device)
                target = target.view(-1, self.feats*self.pred_len).to(self.device)
                
                loss_rec = self.loss(output, target)  
                loss = con_loss - adv_loss + loss_rec + self.lamda* loss_kernel
                
                avg_loss += loss.cpu().item()
                loss.backward()
                self.optimizer.step()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))    
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (input, target) in loop:
                    input = input.float().to(self.device)
                    tematt, output,freatt = self.model(input)
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

                    output = output.view(-1, self.feats*self.pred_len).to(self.device)
                    target = target.view(-1, self.feats*self.pred_len).to(self.device)
                
                    loss_rec = self.loss(output, target)
                    loss = con_loss - adv_loss + loss_rec
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            valid_loss = avg_loss/(max(1, len(valid_loader)))
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
        temperature = 50
        self.model.eval()
        scores = []
        y_hats = []
        attens_energy = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (input, target) in loop:
                input, target = input.float().to(self.device), target.to(self.device)
                tematt, output,freatt = self.model(input)
                output = output.view(-1, self.feats*self.pred_len).to(self.device)
                target = target.view(-1, self.feats*self.pred_len).to(self.device)
                mse = torch.sub(output, target).pow(2)
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')
                

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        assert scores.ndim == 1
        
        # print('scores: ', scores.shape)
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
