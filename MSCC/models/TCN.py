from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                   dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self,
                 n_features,
                 num_channels=[32, 32, 40],
                 kernel_size=3,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 device='cpu'):
        
        super(TCNModel, self).__init__()
        self.n_features = n_features
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.predict_time_steps = predict_time_steps
        self.dropout_rate = dropout_rate
        self.device = device

        # TCN backbone
        self.tcn = TemporalConvNet(num_inputs=n_features,
                                  num_channels=num_channels,
                                  kernel_size=kernel_size,
                                  dropout=dropout_rate)
        
        # Prediction head
        self.linear = nn.Linear(num_channels[-1], n_features)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, n_features)
        x = x.permute(0, 2, 1)  # -> (batch_size, n_features, seq_len)
        y = self.tcn(x)  # Output shape: (batch_size, num_channels[-1], seq_len)
        
        # Take last predict_time_steps outputs
        y = y[:, :, -self.predict_time_steps:]  # (batch_size, num_channels[-1], predict_steps)
        y = y.permute(0, 2, 1)  # -> (batch_size, predict_steps, num_channels[-1])
        
        outputs = self.linear(y)  # (batch_size, predict_steps, n_features)
        return outputs.permute(1, 0, 2)  # (predict_steps, batch_size, n_features)

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
        self.y_hats = None
        self.device = get_gpu(True)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.num_channels = num_channel
        self.lr = lr
        self.validation_size = validation_size
        
        self.model = TCNModel(n_features=feats, 
                            num_channels=num_channel,
                            device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10

    # 保持fit和decision_function方法与原CNN类一致（需要调整内部模型调用）
    def fit(self, data):
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
            self.model.train()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(x)  # 这里调用TCN模型
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)
                
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            # 验证过程保持不变
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    output = output.view(-1, self.feats*self.pred_len)
                    target = target.view(-1, self.feats*self.pred_len)
                    
                    loss = self.loss(output, target)
                    avg_loss += loss.item()
                    # ...其余代码保持不变...
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
        # 测试过程保持不变（需要调整模型调用）
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
                output = self.model(x)  # 这里调用TCN模型
                # ...其余代码保持不变...
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

    # 保持其他辅助方法不变
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))