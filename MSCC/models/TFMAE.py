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
import torchinfo
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import get_gpu


from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset
from ..utils.embed import DataEmbedding, PositionalEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.attn import AttentionLayer
from ..utils.embed import DataEmbedding, PositionalEmbedding

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.00001) - torch.log(q + 0.00001))
    return torch.sum(res, dim=-1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, T, D]
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist

class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, window_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
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

        # converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]

        # masking smaller mag
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]

        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1,2)

        # encoding tokens
        dx, att = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]
    
class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, window_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Encoder(
            [
                    AttentionLayer( d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * window_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        # print("ex:",ex.shape)
        filters = torch.ones(1,1,self.seq_size).to(device)
        ex2 = ex ** 2

        # calculating summation of ex and ex2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr[:,:,self.seq_size-1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr2[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr2[:,:,self.seq_size-1:] /= self.seq_size
        
        # calculating mean and variance
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]
        
        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx = masked_idx)
        
        tokens = torch.zeros(ex.shape,device=device)
        tokens[torch.arange(ex.shape[0])[:,None],unmasked_idx,:] = ux
        tokens[torch.arange(ex.shape[0])[:,None],masked_idx,:] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)
        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, D]
    
class MTFA(nn.Module):  
    def __init__(self, window_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        global device
        device = dev
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, window_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, window_size, fr)

    def forward(self, x):
        tematt = self.tem(x) 
        freatt = self.fre(x)
        # print("freatt:",freatt[0].shape)
        return tematt,freatt
    
   
class TFMAE():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0002,
                 feats=1,
                 d_model = 100,
                 validation_size=0.2):
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

        self.lr = lr
        self.validation_size = validation_size
        
        self.model = MTFA(window_size=self.window_size, seq_size=3, c_in=self.feats, c_out=self.feats, d_model=100, e_layers=3, fr=0.4, tr=0.5,dev=self.device).to(self.device)
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None 
        self.sigma = None
        self.eps = 1e-10
    def vali(self,valid_loader):
        self.model.eval()
        scores = []
        avg_loss = 0
        loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
        with torch.no_grad():
            for idx, (input, target) in loop:
                input = input.float().to(self.device)
                tematt, freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                        freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),tematt[u])))
                    con_loss += (torch.mean(my_kl_loss(
                        (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),tematt[u].detach())) + torch.mean(my_kl_loss(tematt[u].detach(), (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                adv_loss = adv_loss / len(freatt)
                con_loss = con_loss / len(freatt)
                
                loss = con_loss - adv_loss
                # print("con_loss:",con_loss)
                # print("adv_loss:",adv_loss)
                # print("loss_rec:",loss_rec)
                avg_loss += loss.cpu().item()                    
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
        valid_loss = avg_loss/(max(1, len(valid_loader)))
        return valid_loss    
        
        
    def fit(self, data):
        print("tfmae")
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        train_steps = len(train_loader)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (input, target) in loop:
                self.optimizer.zero_grad()
                input, target = input.float().to(self.device), target.to(self.device)
                # print("window_size:",self.window_size)
                # print('x: ', x.shape)       # (bs, win, feat)
                # print('target: ', target.shape)     # # (bs, pred_len, feat)
                # print('len(tsTrain): ', len(tsTrain))
                # print('len(train_loader): ', len(train_loader))

                tematt,freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    # print("---------")
                    # print(tematt[u])
                    # print("--******--")
                    # print(freatt[u])
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
                # print("adv_loss:",adv_loss)
                # print("con_loss:",con_loss)  
                loss = con_loss - adv_loss
                
                avg_loss += loss.cpu().item()
                loss.backward()
                self.optimizer.step()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))    
        
            valid_loss = self.vali(valid_loader)
            self.scheduler.step()
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    
    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        self.model.eval()
        temperature = 50
        scores = []
        y_hats = []
        attens_energy = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (input, _) in loop:
                input = input.float().to(self.device)
                tematt, freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss = my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                    else:
                        adv_loss += my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss += my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                # print("test adv_loss:",adv_loss)
                # print("test con_loss:",con_loss)
                # print("all loss:",adv_loss+con_loss)
                # metric = torch.softmax((adv_loss + con_loss), dim=-1)
               
                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()[:,-1]
                attens_energy.append(cri)
                loop.set_description(f'Testing: ')
                
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        score = np.array(attens_energy)
        scores = np.append(score, score[-1])
        print("scores:",scores.shape)
        assert scores.ndim == 1
        
        import shutil
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.window_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.window_size-1)//2))
        
        return self.__anomaly_score


    def anomaly_score(self) -> np.ndarray:

        if self.__anomaly_score.shape[0] < self.ts_len:
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.window_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.window_size-1)//2))

        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
