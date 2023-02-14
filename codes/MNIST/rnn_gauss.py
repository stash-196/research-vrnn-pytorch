import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the (not Variational) Recurrent
Neural Network (RNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

class RNN(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, bias=False):
        super(RNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        # self.dec_std = nn.Sequential(
        #     nn.Linear(h_dim, x_dim),
        #     nn.Softplus())
        # # correlation and binary for BiGauss
        # self.corr = nn.Sequential(
        #     nn.Linear(h_dim, 1),
        #     nn.Tanh())
        
        # self.binary = nn.Sequential(
        #     nn.Linear(h_dim, 1),
        #     nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(h_dim, h_dim, n_layers, bias)


    def forward(self, x):

        loss = 0

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t])

            #decoder
            dec_t = self.dec(h[-1])
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)
            # dec_corr_t = self.corr(dec_t)
            # dec_binary_t = self.binary(dec_t)

            #recurrence
            _, h = self.rnn(phi_x_t.unsqueeze(0), h) # change this to normal rnn

            #computing losses
            # loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            loss += self._nll_bernoulli(dec_mean_t, x[t])

            assert not loss.isnan()

        return loss


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #decoder
            dec_t = self.dec(h[-1])
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(phi_x_t.unsqueeze(0), h) # change this to normal rnn

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    # for binary data
    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(torch.ones_like(x) *2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))


    def _nll_bigauss(self, x, mean, std, corr, binary):
        """
        Gaussian mixture model negative log-likelihood
        Parameters
        ----------
        y     : Tensor
        mu    : Tensor
        sig   : Tensor
        corr  : Tensor
        binary: Tensor
        """
        # Unsqueeze to convert 1D tensors to 2D tensors with shape (batch_size, 1)
        mean_1 = mean[:, 0].unsqueeze(1)
        mean_2 = mean[:, 1].unsqueeze(1)

        std_1 = std[:, 0].unsqueeze(1)
        std_2 = std[:, 1].unsqueeze(1)

        y0 = x[:, 0].unsqueeze(1)
        y1 = x[:, 1].unsqueeze(1)
        y2 = x[:, 2].unsqueeze(1)
        corr = corr.unsqueeze(1)

        # Calculate the NLL of the binary variable
        binary_term = torch.sum(y0 * binary + (1 - y0) * (1 - binary), dim=1)

        # Constant term that depends only on the parameters of the Gaussian mixture model
        constant_term = torch.log(2 * np.pi) + torch.log(std_1) + torch.log(std_2) + 0.5 * torch.log(1 - corr**2)

        # Calculate the term that depends on the observed data points and the parameters of the Gaussian mixture model
        z = ((y1 - mean_1) / std_1)**2 + ((y2 - mean_2) / std_2)**2 - 2 * corr * (y1 - mean_1) * (y2 - mean_2) / (std_1 * std_2)
        data_term = 0.5 / (1 - corr**2) * z

        # Calculate the overall cost by subtracting the constant and data terms from the binary term
        cost = - constant_term - data_term

        # Sum the cost along the batch dimension and return the negative log-likelihood
        nll = -torch.sum(cost, dim=1) - binary_term
        return nll

