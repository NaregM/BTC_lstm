import numpy as np
import torch
from torch import nn

class LSTM(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, lh):
        
        """
      
        
        """
        
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        
        self.loss_hist = lh
                
        self.lstm = nn.LSTM(input_size, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
        self.hidden_cell = (torch.zeros(1,1, self.hidden_dim), torch.zeros(1,1,self.hidden_dim))
        
        
    def forward(self, input_seq):
        
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        
        predictions = self.fc(lstm_out.view(len(input_seq), -1))
    
        return predictions[-1]
    
    
    @classmethod
    def train_(cls, input_size, output_size, hidden_dim, train_seq, criterion, lh, n_epochs, verbose = True, print_i = 10):
        
        cls_ = cls(input_size, output_size, hidden_dim, lh)
        
        optimizer = torch.optim.Adam(cls_.parameters(), lr = 0.001)

        for i in range(n_epochs):
            
            for seq, label in train_seq:
        
                optimizer.zero_grad()
                
                cls_.hidden_cell = (torch.zeros(1, 1, cls_.hidden_dim),
                                    torch.zeros(1, 1, cls_.hidden_dim))

                y_pred = cls_(seq)

                loss = criterion(y_pred, label)

                loss.backward()
                optimizer.step()
                
            lh.append(loss.item())

            if verbose:
                
                    if i % print_i == 0:

                        print('Epoch', i, ' Loss: ', loss.item())
                              
        return cls_
    
