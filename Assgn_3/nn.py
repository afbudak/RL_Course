import numpy as np
from algo import ValueFunctionWithApproximation

import torch
import torch.nn as nn
from torch.autograd import Variable

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        n_in = state_dims
        n_h = 32
        n_out = 1
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_out))
        
        self.model = self.model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        

    def __call__(self,s):
        # TODO: implement this method
        output = self.model(torch.tensor(s).float()).detach().numpy()
        #return output.tolist()[0]
        return output.tolist()[0]

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        s_tau = torch.tensor(s_tau).float()
        G = torch.tensor([G])
        loss = self.loss_fn(self.model(s_tau), G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None

