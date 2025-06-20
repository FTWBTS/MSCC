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

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class CNNModel(nn.Module):
    def __init__(self,
                 n_features,
                 num_channel=[32, 32, 40],
                 kernel_size=3,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 hidden_activation='relu',
                 device='cpu'):

        # initialize the super class
        super(CNNModel, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.num_channel = num_channel
        self.device = device

        # get the object for the activations functions
        self.activation = get_activation_by_name(hidden_activation)

        # initialize encoder and decoder as a sequential
        self.conv_layers = nn.Sequential()
        prev_channels = self.n_features
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
            # torch.nn.Linear(2*self.num_channel[-1], self.n_features),
            torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.num_channel[-1], self.n_features)
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.view(b, c, l)
        # x = x.permute(0, 2, 1)
        x = self.conv_layers(x)     # [128, feature, 23]
        outputs = torch.zeros(self.predict_time_steps, b, self.n_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
        return outputs
    
class CNN():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 num_channel=[32, 32, 40],
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
        self.num_channel = num_channel
        self.lr = lr
        self.validation_size = validation_size
        
        self.model = CNNModel(n_features=feats, num_channel=num_channel, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        print("调用cnn")
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
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                # print("window_size:",self.window_size)
                # print('x: ', x.shape)       # (bs, win, feat)
                # print('target: ', target.shape)     # # (bs, pred_len, feat)
                # print('len(tsTrain): ', len(tsTrain))
                # print('len(train_loader): ', len(train_loader))

                self.optimizer.zero_grad()
                # print("train_x:",x.shape)
                output = self.model(x)
                # print("train_out",output.shape)
                # print("train_target",target.shape)
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)
                
                loss = self.loss(output, target)
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
                    # print("x.shape",x.shape)
                    output = self.model(x)
                    # print("cnn_output:",output.shape)
                    output = output.view(-1, self.feats*self.pred_len)
                    target = target.view(-1, self.feats*self.pred_len)
                    
                    loss = self.loss(output, target)
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
        
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)

                mse = torch.sub(output, target).pow(2)

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
