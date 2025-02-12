import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 



import sys
import os
FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, '..', '..'))
sys.path.insert(0, ROOT_DIR)
SAVE_DIR = os.path.join(ROOT_DIR, 'saves', 'blizzard')
DATA_DIR = os.path.join(ROOT_DIR, 'data/blizzard/adventure_and_science_fiction')

from codes.blizzard.audio_dataset import AudioDataset, fetch_npy_file_paths


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)


    def forward(self, x):
        """_summary_

        Args:
            x (_type_): entire sequence to train on

        Returns:
            _type_: losses of either kldivervgence or nll-loss
        """
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(x.size(0)): # iterate over time dimension (1st dimension should be the sequence)

            # Forward Propagate the input to get more complex features for
            # every time step.
            phi_x_t = self.phi_x(x[t]) # ToDo: fix float-double problem

            #encoder
            # Generate (infer) the mean and standard deviation of the
            # latent variables Z_t | X_t for every time-step of the LSTM.
            # This is a function of the input ('s extracted features) and 
            # the hidden state of the previous time step.
            # p(z_t|x_t) = N(mean_z_t, std_z_t), [mean_z_t, std_z_t] = phi_enc(phi_x_t, h_t-1)
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #prior
            # Prior on the latent variables at every time-step
            # Dependent only on the hidden-step.
            # p(z_t) = N(mean_0_t, std_0_t), [mean_0_t, std_0_t] = phi_prior(h_t-1)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            # Sample from the latent distibution with mean phi_mu_t
            # and std phi_sig_t
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

            # feature extraction of z_t
            # phi_z(z_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            # Generate the output distribution at every time-step.
            # This is as a function of the latent variables and the hidden-state at
            # every time-step.
            # p(x_t|z_t) = N(mean_x_t, std_x_t), [mean_x_t, std_x_t] = phi_dec(phi_z(z_t), h_t-1)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence
            # h_t = f(phi_z(z_t), phi_x(x_t), h_(t-1)))
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #computing losses
            # KL divergence between variational approximation & prior
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
            assert not (kld_loss.isnan() or nll_loss.isnan())
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)

    # What does this do?
    def sample(self, seq_len):
        """_summary_

        Args:
            seq_len (_type_): _description_

        Returns:
            _type_: _description_
        """
        sample = torch.zeros(seq_len, self.x_dim, device=device) # sequence_length x feature_dim 

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1]) # same as forward
            prior_mean_t = self.prior_mean(prior_t) # same as forward
            prior_std_t = self.prior_std(prior_t) # same as forward

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t) # different from forward() [encoder->prior]
            phi_z_t = self.phi_z(z_t)

            #decoder
            # Generate the output distribution at every time-step.
            # This is as a function of the latent variables and the hidden-state at
            # every time-step.
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

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


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(torch.ones_like(x) *2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))


if __name__ == "__main__":

    # ToDo: change parameters for blizzard
    #hyperparameters
    x_dim = 28
    h_dim = 100
    z_dim = 16
    n_layers = 1
    n_epochs = 25
    clip = 10
    learning_rate = 1e-3
    batch_size = 8 # 128
    seed = 128
    print_every = 1000 # batches
    save_every = 10 # epochs

    #manual seed
    torch.manual_seed(seed)
    plt.ion()

    #init model + optimizer + datasets
    file_paths = fetch_npy_file_paths(DATA_DIR)

    train_loader = torch.utils.data.DataLoader(AudioDataset(file_paths, frame_length=x_dim, train=True), batch_size=batch_size)


    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    for data in train_loader:
        model(data)