import time as tm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from IPython import display
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
import math
from collections import defaultdict
from einops import rearrange
from einops import rearrange
from torch.nn import Parameter
from torch.nn.modules.utils import _pair, _single
import math
from torch.nn.modules.utils import _pair, _single

def kth_diag_indices(a, k):
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)

    return rowidx[:-k], colidx[:-k]


def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x
###########
#  MFVI
###########
class LinearCirculant(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, init_scheme = 'Xavier', bias=True):
        super(LinearCirculant, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Circulant params
        #if kernel_size > (out_features * in_features):
        #    raise Exception("Kernel size must be less or equal to the number of weights") 
        #self.kernel_size = kernel_size 
        
        if kernel_size > (out_features * in_features):
            #raise Exception("Kernel size must be less or equal to the number of weights")
            print("The kernel size extends the weight number o elements, automatically set to " + str(out_features * in_features))
            self.kernel_size = out_features * in_features
        else:
            self.kernel_size = kernel_size 
            
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features * in_features),
                             persistent=False)
        self.init_scheme = init_scheme
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.cholesk_kernel = Parameter(torch.Tensor(self.kernel_size))
        
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        
        self.prior = None
        

    def reset_parameters(self):
        #TODO
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)		
        # Sample choleskies N(0,0.5)
        #self.cholesk_kernel.data.fill_(0.5)
        
        #Normal initialization
        self.cholesk_kernel.data.normal_() 
        
        if self.init_scheme == 'Xavier':
            #self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/( self.in_features * self.out_features))  #Xavier
            #self.cholesk_kernel.data = self.cholesk_kernel.data.fill_(1.00) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            self.cholesk_kernel.data = self.cholesk_kernel.data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
        elif self.init_scheme == 'Scaled':    
            self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/(self.in_features * self.out_features) ) #Scaled
        elif self.init_scheme == '2xScaled':        
            self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/(2*self.in_features * self.out_features) ) #2xScaled
        else:
            raise Exception("Non Implemented Initialization Method") 
        
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        # Forward using the Flipout variance reduction method
        
        #Sample the base perturbation
        eps_weight = self.eps_weight.data.normal_()
        eps_weight_pad = pad_circular_nd(eps_weight[None,:], self.kernel_size-1 , 1)
        #self.cholesk_kernel = torch.flip( self.cholesk_kernel, dims=(0,))#Parameter( )
        delta_weight = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.cholesk_kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight.shape[0]]        
        delta_weight = delta_weight.reshape(self.in_features, self.out_features).T
        
        outputs = F.linear(x, self.weight, self.bias)
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()
        x_tmp = x * sign_input
        

        perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs
        
        return out
    
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        paded_kernel = F.pad(self.cholesk_kernel, (0, self.weight.nelement()- self.kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        

        kld = 0.5*(-self.weight.nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                  torch.sum(self.weight*self.weight) )
        
        #Empirical Bayes
        d_prior = (self.weight.reshape(-1))**2#) + torch.sum(eigs+ 0.001)
        
        log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        trace_prior_inv = torch.prod(d_prior)
        mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        trace_post = torch.sum(eigs)
        log_det_post = torch.sum(torch.log(eigs+0.0000001))
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        
        return kld

    def kl_reg_opt(self):
        
        paded_kernel = F.pad(self.cholesk_kernel, (0, self.weight.nelement()- self.kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        #print(self.prior.shape, self.weight.shape)
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight.T[0:prior.shape[0],:]*self.weight.T[0:prior.shape[0],:] )
        
        trace_post = torch.sum(eigs)
        log_det_post = torch.sum(torch.log(eigs+0.0000001))
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld  
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	
    
    
##############################################################################################################
   
class LinearCirculantFFT(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, bias=True):
        super(LinearCirculantFFT, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Circulant params
        if kernel_size > (out_features * in_features):
            raise Exception("Kernel size must be less or equal to the number of weights") 
        self.kernel_size = kernel_size 
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features * in_features),
                             persistent=False)
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.cholesk_kernel = Parameter(torch.Tensor(kernel_size))
        
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        

    def reset_parameters(self):
        
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)		
        # Sample choleskies N(0,0.5)
        #self.cholesk_kernel.data.fill_(0.5)
        
        #Normal initialization
        #self.cholesk_kernel.data.normal_()#.fill_(0.5)
        #Xavier like intialization
        self.cholesk_kernel.data.normal_() 
        self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/( self.in_features * self.out_features)) 
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        # Forward using the Flipout variance reduction method
        
        spatial_kernel = torch.fft.ifft( torch.exp(self.cholesk_kernel) ).real
        #Sample the base perturbation
        eps_weight = self.eps_weight.data.normal_()
        eps_weight_pad = pad_circular_nd(eps_weight[None,:], self.kernel_size-1 , 1)
        #self.cholesk_kernel = torch.flip( self.cholesk_kernel, dims=(0,))#Parameter( )
        delta_weight = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( spatial_kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight.shape[0]]        
        delta_weight = delta_weight.reshape(self.in_features, self.out_features).T
        
        outputs = F.linear(x, self.weight, self.bias)
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()
        x_tmp = x * sign_input
        

        perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs
        
        return out
    
    def kl_reg(self):

        eigs = self.cholesk_kernel
        #print(paded_kernel)
        #print(torch.min(eigs))
        kld = 0.5*(-self.weight.nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                  torch.sum(self.weight*self.weight) )
        
        return kld

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	    
    
##############################################################################################################    
    
class LinearDiagonal(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearDiagonal, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_var = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features , in_features),
                             persistent=False)        
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        self.prior = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.W.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        #print('Ok')		
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)		
        self.log_var.data.fill_(-5.0)

		
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))
        if var_reducer == 'LRT':
            lrt_mean = F.linear(x, self.weight, self.bias)
            if self.bias is not None:
                lrt_mean = lrt_mean + self.bias

            sigma2 = torch.exp(self.log_var)
            lrt_std = torch.sqrt(1e-16 + F.linear(x * x, sigma2))

            if self.training:
                eps = lrt_std.data.new(lrt_std.size()).normal_()
            else:
                eps = 0.0
            return lrt_mean + lrt_std * eps
        
        
        if var_reducer == 'Flipout':
            eps_weight = self.eps_weight.data.normal_()
            sigma_weight = torch.exp(self.log_var)
            delta_weight = eps_weight*sigma_weight
            delta_weight = delta_weight.reshape(self.in_features, self.out_features).T
            
            outputs = F.linear(x, self.weight, self.bias)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            x_tmp = x * sign_input


            perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs

            return out
            
            
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))		
        kld = 0.5*(-self.weight.nelement() - torch.sum(self.log_var) + torch.sum(torch.exp(self.log_var)) + torch.sum(self.weight*self.weight))
        
        
        
                
        # Empirical Bayes
        d_prior = (self.weight.reshape(-1))**2#) + torch.sum(eigs+ 0.001)
        
        log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        trace_prior_inv = torch.prod(d_prior)
        mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        trace_post = torch.sum(torch.exp(self.log_var))
        log_det_post = torch.sum(self.log_var) 
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        
        return kld
    
    def kl_reg_opt(self):

        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight.T[0:prior.shape[0],:]*self.weight.T[0:prior.shape[0],:] )
        
        trace_post = torch.sum(torch.exp(self.log_var))
        log_det_post = torch.sum(self.log_var) 
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld      
    

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	 

##############################################################################################################

class Conv2dCirculant(nn.Module):

    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                circ_kernel_size,
                init_scheme = 'Xavier',
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv2dCirculant, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.prior = None
 
        self.weight = Parameter( torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        if bias:
            self.bias =  Parameter(torch.zeros( out_channels))
        else:
            self.register_parameter('bias', None)
                
        
        
        # Circulant params
        if circ_kernel_size > (out_channels*in_channels*kernel_size*kernel_size):
            #raise Exception("Kernel size must be less or equal to the number of weights")
            print("The kernel size extends the weight number o elements, automatically set to " + str(out_channels*in_channels*kernel_size*kernel_size))
            self.circ_kernel_size = out_channels*in_channels*kernel_size*kernel_size -10
        else:
            self.circ_kernel_size = circ_kernel_size 
            
        self.cholesk_kernel = Parameter(torch.Tensor(self.circ_kernel_size))
        
        self.register_buffer('eps_weight',
                             torch.Tensor(out_channels*in_channels*kernel_size*kernel_size),
                             persistent=False)
        self.init_scheme = init_scheme
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        
        #Normal initialization
        self.cholesk_kernel.data.normal_() #* np.sqrt( 2/( self.weight.nelement())) 
        #self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/( self.weight.nelement())) #XAVIER
        #self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/( self.weight.nelement())) #Scaled
        #self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/( 2*self.weight.nelement())) #2xScaled
        if self.init_scheme == 'Xavier':
            self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/(  self.weight.nelement()  ))  #Xavier
            #self.cholesk_kernel.data = self.cholesk_kernel.data.fill_(1.00) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            self.cholesk_kernel.data = self.cholesk_kernel.data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            
        elif self.init_scheme == 'Scaled':    
            self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/( self.weight.nelement() ) ) #Scaled
        elif self.init_scheme == '2xScaled':        
            self.cholesk_kernel.data = self.cholesk_kernel.data * ( 1/(2* self.weight.nelement()) ) #2xScaled
        else:
            raise Exception("Non Implemented Initialization Method") 
        if self.bias is not None:
            self.bias.data.zero_()        
        
        if self.bias is not None:
            self.bias.data.zero_()  
            


    def forward(self, x, var_reducer = "Flipout"):
        # Forward using the Flipout variance reduction method
        
        outputs = F.conv2d(x,
                           weight=self.weight,
                           bias=self.bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        eps_weight = self.eps_weight.data.normal_()
        eps_weight_pad = pad_circular_nd(eps_weight[None,:], self.circ_kernel_size-1 , 1)
        #self.cholesk_kernel = torch.flip( self.cholesk_kernel, dims=(0,))#Parameter( )
        #print(eps_weight_pad.shape)
        delta_kernel = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.cholesk_kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight.shape[0]]        
        delta_kernel = delta_kernel.reshape(self.out_channels, self.in_channels // self.groups, *self.kernel_size)        

        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs
        

        
        return out 
    
    
    def kl_reg(self):
        
        paded_kernel = F.pad(self.cholesk_kernel, (0, self.weight.nelement()- self.circ_kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        # KL(q||p), where P = N(0,I)        
        #kld = 0.5*(-self.weight.nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
        #          torch.sum(self.weight*self.weight) )
        
        # KLD where P = N(0, block_diagonal)
        B = np.zeros( [self.kernel_size[0]**2, self.kernel_size[0]**2])
        for i in range(self.kernel_size[0]**2):
            B[kth_diag_indices(B,i)] = np.exp(-np.linalg.norm(i))
            B[kth_diag_indices(B,-i)] = np.exp(-np.linalg.norm(i))
        B = torch.from_numpy(B).float()
        B =  torch.inverse( B + torch.eye(self.kernel_size[0]**2) ).to(self.weight.device)
        
        log_det_prior_inv =  self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (B@self.weight.reshape(self.in_channels * self.out_channels, -1 ).T).reshape(self.weight.shape)*self.weight)
        
        trace_post = torch.sum(eigs)
        log_det_post = torch.sum(torch.log(eigs+0.0000001))
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        
        # Empirical Bayes Diagonal 
        #d_prior = (self.weight.reshape(-1))**2 + 1/torch.sum(eigs+ 0.001)
        #print(d_prior)
        #log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        #trace_prior_inv = torch.prod(d_prior)
        #mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        #trace_post =1/ torch.sum(eigs)
        #log_det_post = torch.sum(torch.log(eigs+0.0000001))
        #kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        
        return kld    

    def kl_reg_opt(self):
        
        paded_kernel = F.pad(self.cholesk_kernel, (0, self.weight.nelement()- self.circ_kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight[0:prior.shape[0],0:prior.shape[1],:]*self.weight[0:prior.shape[0],0:prior.shape[1],:] )
        
        trace_post = torch.sum(eigs)
        log_det_post = torch.sum(torch.log(eigs+0.0000001))
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld  
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
                +', kernel size=' + str(self.kernel_size) +')'   
    
##############################################################################################################

class Conv2dCirculantFFT(nn.Module):

    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                circ_kernel_size,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv2dCirculantFFT, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
 
        self.weight = Parameter( torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        if bias:
            self.bias =  Parameter(torch.zeros( out_channels))
        else:
            self.register_parameter('bias', None)
        self.cholesk_kernel = Parameter(torch.Tensor(circ_kernel_size))        
        
        
        # Circulant params
        if circ_kernel_size > (out_channels*in_channels*kernel_size*kernel_size):
            raise Exception("Kernel size must be less or equal to the number of weights") 
        self.circ_kernel_size = circ_kernel_size 
        self.register_buffer('eps_weight',
                             torch.Tensor(out_channels*in_channels*kernel_size*kernel_size),
                             persistent=False)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        

        self.cholesk_kernel.data.normal_() #* np.sqrt( 2/( self.weight.nelement())) 
        self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/( self.weight.nelement())) 
        if self.bias is not None:
            self.bias.data.zero_()        
        
        if self.bias is not None:
            self.bias.data.zero_()  
            


    def forward(self, x, var_reducer = "Flipout"):
        # Forward using the Flipout variance reduction method
        
        outputs = F.conv2d(x,
                           weight=self.weight,
                           bias=self.bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        spatial_kernel = torch.fft.ifft( torch.exp(self.cholesk_kernel) ).real
        eps_weight = self.eps_weight.data.normal_()
        eps_weight_pad = pad_circular_nd(eps_weight[None,:], self.circ_kernel_size-1 , 1)
        #self.cholesk_kernel = torch.flip( self.cholesk_kernel, dims=(0,))#Parameter( )
        delta_kernel = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( spatial_kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight.shape[0]]        
        delta_kernel = delta_kernel.reshape(self.out_channels, self.in_channels // self.groups, *self.kernel_size)        

        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs
        

        
        return out 
    
    
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        spatial_kernel = torch.fft.ifft( torch.exp(self.cholesk_kernel) ).real        
        paded_spatial_kernel = F.pad(spatial_kernel, (0, self.weight.nelement()- self.circ_kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_spatial_kernel)).real**2
        #print(torch.min(eigs))
        kld = 0.5*(-self.weight.nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                  torch.sum(self.weight*self.weight) )
        
        return kld    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
                +', kernel size=' + str(self.kernel_size) +')'   
    
##############################################################################################################
  
    
class Conv2dDiagonal(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv2dDiagonal, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        self.log_var = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        self.register_buffer('eps_kernel',
                             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size),
                             persistent=False)        
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.prior = None

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.log_var.data.fill_(-5.0)
        #torch.nn.init.normal_(self.log_var, mean=-4, std=0.1)
        if self.bias is not None:
            self.bias.data.zero_()  


    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))
        if var_reducer == 'LRT':
            sampled_output = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            epsilon_var = torch.randn(sampled_output.shape, device=x.device)
            sampled_output += epsilon_var * torch.sqrt(
                F.conv2d(
                    x ** 2,
                    torch.exp(self.log_var),
                    self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )


            return sampled_output
        elif var_reducer == 'Flipout':
            outputs = F.conv2d(x,
                               weight=self.weight,
                               bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.exp(self.log_var)
            eps_kernel = self.eps_kernel.data.normal_()
            delta_kernel = (sigma_weight * eps_kernel)



            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.bias,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            return out    
                        
        else:
            raise Exception("Variance reducer must be either 'LRT' or 'Flipout' ") 

    def kl_reg_opt(self):
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight[0:prior.shape[0],0:prior.shape[1],:]*self.weight[0:prior.shape[0],0:prior.shape[1],:] )
        
        trace_post = torch.sum(torch.exp(self.log_var)) 
        log_det_post = torch.sum(self.log_var)
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld
        
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #kld = 0.5*(-self.weight.nelement() - torch.sum(self.log_var) + 
        #           torch.sum(torch.exp(self.log_var)) + torch.sum(self.weight*self.weight))
        
        
        # KLD where P = N(0, block_diagonal)
        B = np.zeros( [self.kernel_size[0]**2, self.kernel_size[0]**2])
        for i in range(self.kernel_size[0]**2):
            B[kth_diag_indices(B,i)] = np.exp(-np.linalg.norm(i))
            B[kth_diag_indices(B,-i)] = np.exp(-np.linalg.norm(i))
        B = torch.from_numpy(B).float()
        B =  torch.inverse( B + torch.eye(self.kernel_size[0]**2) ).to(self.weight.device)
        
        log_det_prior_inv =  self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (B@self.weight.reshape(self.in_channels * self.out_channels, -1 ).T).reshape(self.weight.shape)*self.weight)
        
        trace_post = torch.sum(torch.exp(self.log_var))
        log_det_post = torch.sum(self.log_var) 
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        
        
        # Empirical Bayes
        #d_prior = (self.weight.reshape(-1))**2#) + torch.sum(eigs+ 0.001)
        
        #log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        #trace_prior_inv = torch.prod(d_prior)
        #mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        #trace_post = torch.sum(torch.exp(self.log_var))
        #log_det_post = torch.sum(self.log_var) 
        #kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        
        return kld    
    
##############################################################################################################


class Conv1dCirculant(nn.Module):

    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                circ_kernel_size,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv1dCirculant, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size# _pair(kernel_size)
        self.padding = padding#_pair(padding)
        self.dilation = dilation#_pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
 
        self.weight = Parameter( torch.Tensor(out_channels, in_channels // self.groups,  self.kernel_size))
        if bias:
            self.bias =  Parameter(torch.zeros( out_channels))
        else:
            self.register_parameter('bias', None)
        self.cholesk_kernel = Parameter(torch.Tensor(circ_kernel_size))        
        
        
        # Circulant params
        if circ_kernel_size > (out_channels*in_channels*kernel_size*kernel_size):
            raise Exception("Kernel size must be less or equal to the number of weights") 
        self.circ_kernel_size = circ_kernel_size 
        self.register_buffer('eps_weight',
                             torch.Tensor(out_channels*in_channels*kernel_size),
                             persistent=False)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.cholesk_kernel.data.normal_()#.fill_(0.5)
        if self.bias is not None:
            self.bias.data.zero_()  
            


    def forward(self, x, var_reducer = "Flipout"):
        # Forward using the Flipout variance reduction method
        
        outputs = F.conv1d(x,
                           weight=self.weight,
                           bias=self.bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        eps_weight = self.eps_weight.data.normal_()
        eps_weight_pad = pad_circular_nd(eps_weight[None,:], self.circ_kernel_size-1 , 1)
        #self.cholesk_kernel = torch.flip( self.cholesk_kernel, dims=(0,))#Parameter( )
        delta_kernel = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.cholesk_kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight.shape[0]]        
        delta_kernel = delta_kernel.reshape(self.out_channels, self.in_channels // self.groups, self.kernel_size)        

        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv1d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs
        

        
        return out 
    
    
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        paded_kernel = F.pad(self.cholesk_kernel, (0, self.weight.nelement()- self.circ_kernel_size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        kld = 0.5*(-self.weight.nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                  torch.sum(self.weight*self.weight) )
        
        return kld    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
                +', kernel size=' + str(self.kernel_size) +')'   
    
##############################################################################################################
    
    
class Conv1dDiagonal(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv1dDiagonal, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size#_pair(kernel_size)
        self.padding = padding#_pair(padding)
        self.dilation = dilation#_pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, self.kernel_size), requires_grad=True)
        self.log_var = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, self.kernel_size), requires_grad=True)
        self.register_buffer('eps_kernel',
                             torch.Tensor(out_channels, in_channels // self.groups, self.kernel_size),
                             persistent=False)        
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.prior = None
        

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.log_var.data.normal_()#.fill_(0.5)
        if self.bias is not None:
            self.bias.data.zero_()  


    def forward(self, x, var_reducer = "Flipout"):
        if var_reducer == 'LRT':
            sampled_output = F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            epsilon_var = torch.randn(sampled_output.shape, device=x.device)
            sampled_output += epsilon_var * torch.sqrt(
                F.conv1d(
                    x ** 2,
                    torch.exp(self.log_var),
                    self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )


            return sampled_output
        elif var_reducer == 'Flipout':
            outputs = F.conv1d(x,
                               weight=self.weight,
                               bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.exp(self.log_var)
            eps_kernel = self.eps_kernel.data.normal_()
            delta_kernel = (sigma_weight * eps_kernel)



            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv1d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.bias,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            return out    
                        
        else:
            raise Exception("Variance reducer must be either 'LRT' or 'Flipout' ") 

    def kl_reg_opt(self):
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight[0:prior.shape[0],0:prior.shape[1],:]*self.weight[0:prior.shape[0],0:prior.shape[1],:] )
        
        trace_post = torch.sum(torch.exp(self.log_var)) 
        log_det_post = torch.sum(self.log_var)
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld  
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        kld = 0.5*(-self.weight.nelement() - torch.sum(self.log_var) + 
                   torch.sum(torch.exp(self.log_var)) + torch.sum(self.weight*self.weight))
        return kld    
    
    
##############################################################################################################
    
class LinearKTied(nn.Module):

    def __init__(self, in_features, out_features, k = 2, init_log_var = -5.0, bias=True):
        super(LinearKTied, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_var_u = Parameter(torch.Tensor(out_features, k))
        self.log_var_v = Parameter(torch.Tensor(in_features, k))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features , in_features),
                             persistent=False)        
        self.init_log_var = init_log_var
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.W.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        #print('Ok')		
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)		
        self.log_var_u.data.fill_(self.init_log_var)
        self.log_var_v.data.fill_(self.init_log_var)
        
        #self.log_var_v.data = self.log_var_v.data.normal_() * self.init_log_var
        #self.log_var_u.data = self.log_var_u.data.normal_() * self.init_log_var
        
        #self.log_var_v.data =  self.init_log_var + self.log_var_v.data.normal_() * 0.1
        #self.log_var_u.data =  self.init_log_var + self.log_var_u.data.normal_() * 0.1
        
        #self.log_var_u.data.normal_()#.fill_(0.5)
        #self.log_var_v.data.normal_()#.fill_(0.5)
        #self.cholesk_kernel.data.normal_() 
        #self.cholesk_kernel.data = self.cholesk_kernel.data * np.sqrt( 2/( self.in_features * self.out_features))         
        #self.log_var.data.fill_(0.0)
		
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        
        if var_reducer == 'LRT':
            lrt_mean = F.linear(x, self.weight, self.bias)
            if self.bias is not None:
                lrt_mean = lrt_mean + self.bias

            sigma2 = torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T

            #sigma2 = torch.exp(log_var)
            lrt_std = torch.sqrt(1e-16 + F.linear(x * x, sigma2))

            if self.training:
                eps = lrt_std.data.new(lrt_std.size()).normal_()
            else:
                eps = 0.0
            return lrt_mean + lrt_std * eps
        
        
        if var_reducer == 'Flipout':
            eps_weight = self.eps_weight.data.normal_()
            #log_var = self.log_var_u @ self.log_var_v.T
            sigma_weight = torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T
            
            #sigma_weight = torch.exp(log_var)
            delta_weight = eps_weight*sigma_weight
            delta_weight = delta_weight.reshape(self.in_features, self.out_features).T
            
            outputs = F.linear(x, self.weight, self.bias)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            x_tmp = x * sign_input


            perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs

            return out
            
            
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))		
        #log_var = self.log_var_u @ self.log_var_v.T
        log_var = torch.log( torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T )
        
        kld = 0.5*(-self.weight.nelement() - torch.sum(log_var) + torch.sum(torch.exp(log_var)) + torch.sum(self.weight*self.weight))
        return kld

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	 	
    
##############################################################################################################
    
    
class Conv2dKtied(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding = 0,
                dilation = 1,
                groups = 1,
                k = 2,
                init_log_var = -5.0, 
                bias = False):
        
        super(Conv2dKtied, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        #self.log_var = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        self.log_var_u = Parameter(torch.Tensor(out_channels, k))
        self.log_var_v = Parameter(torch.Tensor( in_channels * kernel_size * kernel_size, k))    
        self.init_log_var = init_log_var
        
        self.register_buffer('eps_kernel',
                             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size),
                             persistent=False)        
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.log_var_u.data.fill_(self.init_log_var)
        self.log_var_v.data.fill_(self.init_log_var)   
        
        #self.log_var_v.data = self.log_var_v.data.normal_() * self.init_log_var
        #self.log_var_u.data = self.log_var_u.data.normal_() * self.init_log_var
        #self.log_var_v.data =  self.init_log_var + self.log_var_v.data.normal_() * 0.1
        #self.log_var_u.data =  self.init_log_var + self.log_var_u.data.normal_() * 0.1
        
        
        if self.bias is not None:
            self.bias.data.zero_()  


    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))
        if var_reducer == 'LRT':
            sampled_output = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            epsilon_var = torch.randn(sampled_output.shape, device=x.device)
            sigma_weight = (torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T).reshape(self.eps_kernel.shape)
            sampled_output += epsilon_var * torch.sqrt(
                F.conv2d(
                    x ** 2,
                    sigma_weight,
                    self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )


            return sampled_output
        elif var_reducer == 'Flipout':
            outputs = F.conv2d(x,
                               weight=self.weight,
                               bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = (torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T).reshape(self.eps_kernel.shape)
            #sigma_weight = torch.exp(self.log_var)
            eps_kernel = self.eps_kernel.data.normal_()
            delta_kernel = (sigma_weight * eps_kernel)



            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.bias,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            return out    
                        
        else:
            raise Exception("Variance reducer must be either 'LRT' or 'Flipout' ") 

    
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        log_var = torch.log( torch.exp( self.log_var_u) @ torch.exp(self.log_var_v).T )        
        kld = 0.5*(-self.weight.nelement() - torch.sum(log_var) + 
                   torch.sum(torch.exp(log_var)) + torch.sum(self.weight*self.weight))
        return kld      
    
##############################################################################################################

class LinearELRG(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_posterior_var: float = 0.006737946999085467,# 1e-6,
        bias: bool = True,
        rank: int = 1,
        alpha: float = None,
        train_var=True,
        var_reducer = 'Flipout',
    ):
        super().__init__() 
        self.var_reducer = var_reducer
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.rank = rank
        self.alpha = 1/rank
        self.sqrt_alpha = torch.sqrt(torch.tensor(self.alpha))
        self.initial_posterior_var = initial_posterior_var
        
        dims_mean = (out_features, in_features)
        self.w_posterior_mean    = nn.Parameter(torch.Tensor(*dims_mean), requires_grad=True)
        self.w_posterior_v       = nn.Parameter(torch.Tensor(self.rank, *dims_mean), requires_grad=True)
        self.w_posterior_log_var = nn.Parameter(torch.Tensor(*dims_mean), requires_grad=True)
        

        # Use bias
        if bias:
            #self.bias = Parameter(torch.Tensor(1, out_features))
            dims_bias = (out_features,)
            self.bias    = nn.Parameter(torch.Tensor(*dims_bias), requires_grad=True)

        else:
            self.register_parameter('bias', None)
            


        #self.bias = bias
        #if self.bias:
            
            
        self.reset_parameters()
        
        self.weight = self.w_posterior_mean
        
        if not train_var:
            self.w_posterior_log_var.requires_grad = False
            
        self.prior = None
        
    def reset_parameters(self):
        #torch.nn.init.xavier_normal_(self.w_posterior_mean, gain=1.0)
        torch.nn.init.xavier_uniform_(self.w_posterior_mean)
        torch.nn.init.xavier_normal_(self.w_posterior_v, gain=0.5)
        torch.nn.init.constant_(self.w_posterior_log_var, np.log(self.initial_posterior_var))
        torch.nn.init.xavier_normal_(self.w_posterior_v, gain=0.5)
        #self.w_posterior_v.data.fill_(0.00673794)  #INIT2
        if self.use_bias:
            torch.nn.init.normal_(self.bias, mean=0, std=0.1)
            self.bias.data.zero_()      
            
    def forward(self, x, num_samples=0):
        
        if self.var_reducer == 'LRT':
            if num_samples:
                y = []
                for s in range(num_samples):
                    y.append(self.forward(x))
                return torch.cat(y)

            #x = x/(np.sqrt(x.shape[1]))		
            epsilon_var = torch.randn((x.shape[0], self.out_features), device=x.device)
            epsilon_v = torch.randn((x.shape[0], self.rank), device=x.device)

            sampled_output = F.linear(x, self.w_posterior_mean, self.bias)

            sampled_output += epsilon_var * torch.sqrt(
                F.linear(
                    x ** 2 ,
                    torch.exp(self.w_posterior_log_var),
                    None,
                )
            + 0.00001)
            v_output = torch.zeros_like(sampled_output)

            for k in range(self.rank):
                v_output += epsilon_v[:, k].unsqueeze(dim=-1) * F.linear(
                    x,
                    self.w_posterior_v[k],
                    None,
                )

            return sampled_output + self.sqrt_alpha * v_output
        
        if self.var_reducer == 'Flipout':
            sampled_output = F.linear(x, self.w_posterior_mean, self.bias)
            
            eps_weight_diag  = torch.randn( self.w_posterior_log_var.shape, device=x.device)
            sigma_weight_diag = torch.exp(self.w_posterior_log_var)
            
            
            epsilon_v = torch.randn((self.rank), device=x.device)
            delta_weight =  sigma_weight_diag * eps_weight_diag    
            delta_weight =  delta_weight + self.alpha*(self.w_posterior_v.reshape(self.rank,-1).T @ epsilon_v).reshape(sigma_weight_diag.shape) 
            delta_weight = delta_weight.reshape(self.in_features, self.out_features).T
            
            outputs = F.linear(x, self.w_posterior_mean, self.bias)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            x_tmp = x * sign_input


            perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            
            return out
            #torch.exp(self.w_posterior_log_var)
    
    def kl_reg_opt(self):
        prior_var = 1.0#np.exp(-1.0)
        device    = self.w_posterior_log_var.device
        means     = torch.flatten(self.w_posterior_mean)
        vs        = torch.flatten(self.w_posterior_v,1)
        log_vars  = torch.flatten(self.w_posterior_log_var)
        vars      = torch.exp(log_vars)
        alpha     = self.alpha
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.w_posterior_mean.T[0:prior.shape[0],:]*self.w_posterior_mean.T[0:prior.shape[0],:] )
        
        trace_post = torch.sum(vars) + (torch.norm(vs, 2, dim=1) ** 2).sum()
        log_det_post = torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        
        kld = 0.5*( -self.w_posterior_mean.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld     
    
    def kl_reg(self):
        
        prior_var = 1.0#np.exp(-1.0)
        device    = self.w_posterior_log_var.device
        means     = torch.flatten(self.w_posterior_mean)
        vs        = torch.flatten(self.w_posterior_v,1)
        log_vars  = torch.flatten(self.w_posterior_log_var)
        vars      = torch.exp(log_vars)
        alpha     = self.alpha

        # Calculate KL for individual normal distributions over parameters
        kl = (vars / prior_var - log_vars).sum()
        kl += (torch.norm(vs, 2, dim=1) ** 2).sum() * alpha / prior_var
        kl -= torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        
        #kl += torch.norm(means, 2) ** 2 / prior_var
        #kl += np.log(prior_var) - 1.0
        
        
        kl += torch.norm(means, 2) ** 2 / prior_var
        kl += torch.sum(self.w_posterior_mean*self.w_posterior_mean)
        kl += np.log(prior_var) - 1.0 - self.w_posterior_mean.nelement()
        return 0.5 * kl 
    
    
        # Empirical Bayes
        #d_prior = (self.weight.reshape(-1))**2 #+ torch.sum(eigs+ 0.001)
        
        #log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        #trace_prior_inv = torch.prod(d_prior)
        #mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        #trace_post = torch.sum(vars) + (torch.norm(vs, 2, dim=1) ** 2).sum()
        #log_det_post = torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        #kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        #return kld
    
        """
        ##########################################
        # Only for k=1
        ##########################################
        kl = 0
        size = 1000#100#20
        #kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.ones(size).to(self.w_posterior_log_var.device)
        kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.randn(size).to(self.w_posterior_log_var.device)
        #kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.arange(size).to(self.w_posterior_log_var.device)
        
        
        log_vars  = torch.flatten(self.w_posterior_log_var)
        v = self.w_posterior_v.reshape(self.rank,-1) 
        
        
        # Apply Empirical Bayes #        
        ############################################################################################
        #m = self.weight.reshape(-1)[0:size]
        #S = torch.diag(torch.exp(log_vars)[0:size]) + self.w_posterior_v.reshape(self.rank,-1)[:,0:size].T @  self.w_posterior_v.reshape(self.rank,-1)[:,0:size] 
        #P = torch.diag(torch.ones(size-1), diagonal=1).to(self.w_posterior_log_var.device)
        #P[size-1,0]=1.0
        
        #for i in range(size):
        #    if i == 0:
        #        kernel[i] = torch.trace(  S )  + m.T@m
        #    else:
        #        kernel[i] = torch.trace( P.pow(i) @ S )  + m@P.pow(i)@m
        ##kernel = 0.0001*kernel       
        #kernel = 0.00001*kernel       
        ############################################################################################
        paded_kernel = F.pad(kernel, (0, self.weight.nelement()- size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        kl = kl -  (   torch.log(1 + torch.sum(( v* (1.0/(torch.exp(log_vars)+ 0.00001)  ) )*v)) + torch.sum(log_vars)   )
        kl = kl -  self.w_posterior_mean.nelement()
        kl = kl -  torch.sum(torch.log(eigs+0.0000001))
        v_ =  pad_circular_nd(v, size-1 , 1)
        delta_weight = torch.nn.functional.conv1d(  v_[None,:]   ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:v.shape[0]]#[0,0:v.shape[0]]  
        kl = kl +  torch.sum(self.w_posterior_mean*delta_weight)

        v_1 = pad_circular_nd(torch.exp(log_vars).reshape(1,-1), size-1 , 1) 
        delta_weight = torch.nn.functional.conv1d(   v_1[None,:]    ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0:log_vars.shape[0]]  
        kl = kl + torch.sum(delta_weight)
       
        
        v_2 = pad_circular_nd(self.w_posterior_mean.reshape(1,-1) , size-1 , 1) 
        delta_weight_rank = torch.nn.functional.conv1d( v_2[None,:] ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0:self.w_posterior_mean.reshape(1,-1).shape[1]]        
        #print(self.w_posterior_mean.reshape(-1).shape)
        #print(delta_weight_rank.shape)
        kl = kl + torch.sum(delta_weight_rank*self.w_posterior_mean.reshape(-1)[0:delta_weight_rank.shape[0]] )
        
        return 0.5*kl  
        """
    
##############################################################################################################

class Conv2dELRG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        initial_posterior_var: float = 0.006737946999085467,#1e-6,
        rank: int = 1,
        alpha: float = None,
        var_reducer = 'Flipout',
        *args,
        **kwargs,
    ):
        super().__init__()
        self.var_reducer = var_reducer
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.rank = rank
        self.alpha = 1/rank
        self.sqrt_alpha = torch.sqrt(torch.tensor(alpha))
        self.initial_posterior_var = initial_posterior_var
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels        
        self.prior = None
        (
            self.w_posterior_mean,
            self.w_posterior_v,
            self.w_posterior_log_var,
        ) = self.create_parameter(
            "weight", (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            (
                self.b_posterior_mean,
                self.b_posterior_v,
                self.b_posterior_log_var,
            ) = self.create_parameter("bias", (out_channels,))
        else:
            self.register_parameter("b_posterior_mean", None)
            self.register_parameter("b_posterior_v", None)
            self.register_parameter("b_posterior_log_var", None)
        self.reset_parameters()

        # to be compatible with deterministic model
        self.weight = self.w_posterior_mean
        if bias:
            self.bias = self.b_posterior_mean

        # To freeze var:
        self.w_posterior_log_var.requires_grad = False
        if bias:
            self.b_posterior_log_var.requires_grad = False

    def create_parameter(self, name, dims):
        posterior_mean = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        posterior_v = nn.Parameter(torch.Tensor(self.rank, *dims), requires_grad=True)
        posterior_log_var = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        return posterior_mean, posterior_v, posterior_log_var

    def reset_parameters(self):
        #torch.nn.init.xavier_normal_(self.w_posterior_mean, gain=1.0)
        torch.nn.init.kaiming_normal_(self.w_posterior_mean, mode="fan_out", nonlinearity="relu")
        
        torch.nn.init.xavier_normal_(self.w_posterior_v, gain=0.5)
        # Initialise the posterior variances with the given constant value.
        torch.nn.init.constant_(
            self.w_posterior_log_var, math.log(self.initial_posterior_var)
        )
        #self.w_posterior_v.data.fill_(0.00673794)
        if self.use_bias:
            torch.nn.init.normal_(self.b_posterior_mean, mean=0, std=0.1)
            torch.nn.init.normal_(self.b_posterior_v, mean=0, std=0.1)
            # Initialise the posterior variances with the given constant value.
            torch.nn.init.constant_(
                self.b_posterior_log_var, math.log(self.initial_posterior_var)
            )
            self.b_posterior_mean.data.zero_()      

    def forward(self, x, num_samples=0):
        if self.var_reducer == 'LRT':
            if num_samples:
                y = []
                for s in range(num_samples):
                    y.append(self.forward(x))
                return torch.cat(y)
            sampled_output = F.conv2d(
                x,
                self.w_posterior_mean,
                self.b_posterior_mean,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            epsilon_var = torch.randn(sampled_output.shape, device=x.device)
            sampled_output += epsilon_var * torch.sqrt(
                F.conv2d(
                    x ** 2,
                    torch.exp(self.w_posterior_log_var),
                    torch.exp(self.b_posterior_log_var) if self.use_bias else None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )

            epsilon_v = torch.randn((x.shape[0], self.rank), device=x.device)
            v_output = torch.zeros_like(sampled_output)
            for k in range(self.rank):
                epsilon_v_k = epsilon_v[:, k][(...,) + (None,) * 3]  # unsqueeze last 3 dims
                v_output += epsilon_v_k * F.conv2d(
                    x,
                    self.w_posterior_v[k],
                    self.b_posterior_v[k] if self.use_bias else None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )

            return sampled_output + self.sqrt_alpha * v_output
        
        if self.var_reducer == 'Flipout':
            outputs = F.conv2d(x,
                               weight=self.w_posterior_mean,
                               bias=self.b_posterior_mean,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            
            
            eps_weight_diag  = torch.randn( self.w_posterior_log_var.shape, device=x.device)
            sigma_weight_diag = torch.exp(self.w_posterior_log_var)
            
            
            epsilon_v = torch.randn((self.rank), device=x.device)
            delta_weight =  sigma_weight_diag * eps_weight_diag    
            delta_weight =  delta_weight + self.alpha*(self.w_posterior_v.reshape(self.rank,-1).T @ epsilon_v).reshape(sigma_weight_diag.shape) 
            delta_kernel = delta_weight.reshape(self.w_posterior_mean.shape)
            
            
            
            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.b_posterior_mean,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            return out    

    def kl_reg_opt(self):
        prior_var = 1.0#np.exp(-1.0)
        device    = self.w_posterior_log_var.device
        means     = torch.flatten(self.w_posterior_mean)
        vs        = torch.flatten(self.w_posterior_v,1)
        log_vars  = torch.flatten(self.w_posterior_log_var)
        vars      = torch.exp(log_vars)
        alpha     = self.alpha
        
        prior = self.prior**2
        log_det_prior_inv =  torch.sum( torch.log( prior + 0.01)  )   #self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = torch.sum(1/(prior + 0.0001))#self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (1/(prior+0.0001))*self.weight[0:prior.shape[0],0:prior.shape[1],:]*self.weight[0:prior.shape[0],0:prior.shape[1],:] )
        
        trace_post = torch.sum(vars) + (torch.norm(vs, 2, dim=1) ** 2).sum()
        log_det_post = torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        return kld
    def kl_reg(self):
        
        prior_var = 1.0#np.exp(-1.0)
        device    = self.w_posterior_log_var.device
        means     = torch.flatten(self.w_posterior_mean)
        vs        = torch.flatten(self.w_posterior_v,1)
        log_vars  = torch.flatten(self.w_posterior_log_var)
        vars      = torch.exp(log_vars)
        alpha     = self.alpha
        """
        # Calculate KL for individual normal distributions over parameters
        kl = (vars / prior_var - log_vars).sum()
        kl += (torch.norm(vs, 2, dim=1) ** 2).sum() * alpha / prior_var
        kl -= torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        
        #kl += torch.norm(means, 2) ** 2 / prior_var
        kl += torch.sum(self.w_posterior_mean*self.w_posterior_mean)
        kl += np.log(prior_var) - 1.0 - self.w_posterior_mean.nelement()
        #return 0.5 * kl 
        """
        
        # KLD where P = N(0, block_diagonal)
        B = np.zeros( [self.kernel_size[0]**2, self.kernel_size[0]**2])
        for i in range(self.kernel_size[0]**2):
            B[kth_diag_indices(B,i)] = np.exp(-np.linalg.norm(i))
            B[kth_diag_indices(B,-i)] = np.exp(-np.linalg.norm(i))
        B = torch.from_numpy(B).float()
        B =  torch.inverse( B + torch.eye(self.kernel_size[0]**2) ).to(self.weight.device)
        
        log_det_prior_inv =  self.kernel_size[0]**2 * torch.logdet(B)
        trace_prior_inv = self.in_channels * self.out_channels * B[0,0] * self.kernel_size[0]**2
        mean_times_prior_inv = torch.sum( (B@self.weight.reshape(self.in_channels * self.out_channels, -1 ).T).reshape(self.weight.shape)*self.weight)
        
        trace_post = torch.sum(vars) + (torch.norm(vs, 2, dim=1) ** 2).sum()
        log_det_post = torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        
        kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)
        
        
        
        # Empirical Bayes
        #d_prior = (self.weight.reshape(-1))**2#) + torch.sum(eigs+ 0.001)
        
        #log_det_prior_inv =  torch.log(torch.prod(d_prior) + 0.0001)
        #trace_prior_inv = torch.prod(d_prior)
        #mean_times_prior_inv = torch.sum( (d_prior * self.weight.reshape(-1 )).reshape(self.weight.shape)*self.weight)
        
        #trace_post = torch.sum(vars) + (torch.norm(vs, 2, dim=1) ** 2).sum()
        #log_det_post = torch.logdet( torch.eye(self.rank, device=device) + alpha * ((vs / vars) @ vs.T) )
        #kld = 0.5*( -self.weight.nelement()   - log_det_prior_inv - log_det_post + trace_post*trace_prior_inv + mean_times_prior_inv)  
        
        return kld
        
        """
        ##########################################
        # Only for k=1
        ##########################################
        kl = 0
        size = 1000#100#20 
        #kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.ones(size).to(self.w_posterior_log_var.device)
        kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.randn(size).to(self.w_posterior_log_var.device)
        #kernel = np.sqrt( 2/(  self.weight.nelement()  ))   *  torch.arange(size).to(self.w_posterior_log_var.device)
        
        
        log_vars  = torch.flatten(self.w_posterior_log_var)
        v = self.w_posterior_v.reshape(self.rank,-1) 
        
        
        # Apply Empirical Bayes #        
        ############################################################################################
        
        #m = self.weight.reshape(-1)[0:size]
        #S = torch.diag(torch.exp(log_vars)[0:size]) + self.w_posterior_v.reshape(self.rank,-1)[:,0:size].T @  self.w_posterior_v.reshape(self.rank,-1)[:,0:size] 
        #P = torch.diag(torch.ones(size-1), diagonal=1).to(self.w_posterior_log_var.device)
        #P[size-1,0]=1.0
        
        #for i in range(size):
        #    if i == 0:
        #        kernel[i] = torch.trace(  S )  + m.T@m
        #    else:
        #        kernel[i] = torch.trace( P.pow(i) @ S )  + m@P.pow(i)@m
        ##kernel = 0.0001*kernel       
        #kernel = 0.00001*kernel       
        ############################################################################################
        paded_kernel = F.pad(kernel, (0, self.weight.nelement()- size), 'constant')
        eigs = (torch.fft.fft(paded_kernel)).real**2
        
        kl = kl -  (   torch.log(1 + torch.sum(( v* (1.0/(torch.exp(log_vars)+ 0.00001)  ) )*v)) + torch.sum(log_vars)   )
        kl = kl -  self.w_posterior_mean.nelement()
        kl = kl -  torch.sum(torch.log(eigs+0.0000001))
        v_ =  pad_circular_nd(v, size-1 , 1)
        delta_weight = torch.nn.functional.conv1d(  v_[None,:]   ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:v.shape[0]]#[0,0:v.shape[0]]  
        kl = kl +  torch.sum(self.w_posterior_mean*delta_weight)

        v_1 = pad_circular_nd(torch.exp(log_vars).reshape(1,-1), size-1 , 1) 
        delta_weight = torch.nn.functional.conv1d(   v_1[None,:]    ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0:log_vars.shape[0]]  
        kl = kl + torch.sum(delta_weight)
       
        
        v_2 = pad_circular_nd(self.w_posterior_mean.reshape(1,-1) , size-1 , 1) 
        delta_weight_rank = torch.nn.functional.conv1d( v_2[None,:] ,
                                        torch.flip( kernel, dims=(0,))[None,None,:], 
                                        padding=0)[0,0:self.w_posterior_mean.reshape(1,-1).shape[1]]        
        #print(self.w_posterior_mean.reshape(-1).shape)
        #print(delta_weight_rank.shape)
        kl = kl + torch.sum(delta_weight_rank*self.w_posterior_mean.reshape(-1)[0:delta_weight_rank.shape[0]] )
        
        return 0.5*kl
        """
        
        
class LinearRank1Circulant(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, components, bias=True, init_scheme = 'Xavier'):
        super(LinearRank1Circulant, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        if kernel_size > min(out_features , in_features):
            #raise Exception("Kernel size must be less or equal to the number of weights")
            print("The kernel size extends the weight number o elements, automatically set to " + str(out_features * in_features))
            self.kernel_size = min(out_features , in_features)
        else:
            self.kernel_size = kernel_size 
        
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
 
        self.s_mean    = nn.ParameterList( Parameter(torch.Tensor(in_features)) for _ in range(components))
        self.s_cholesk_kernel =  nn.ParameterList( Parameter(torch.Tensor(self.kernel_size)) for _ in range(components)) 
        
        self.r_mean    = nn.ParameterList( Parameter(torch.Tensor(out_features)) for _ in range(components))
        self.r_cholesk_kernel =  nn.ParameterList( Parameter(torch.Tensor(self.kernel_size)) for _ in range(components))
        
        self.register_buffer('eps_weight_s', torch.Tensor(in_features), persistent=False) 
        self.register_buffer('eps_weight_r', torch.Tensor(out_features), persistent=False)  
              
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.components = components
        self.component_counter = 0        
        self.reset_parameters()


    def reset_parameters(self): #TODO
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.W.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        #print('Ok')		
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)	
        for i in range(self.components):
            self.s_cholesk_kernel[i].data.fill_(-5.0).to(self.weight.device)
            self.r_cholesk_kernel[i].data.fill_(-5.0).to(self.weight.device)
            
            self.s_cholesk_kernel[i].data = self.s_cholesk_kernel[i].data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            self.r_cholesk_kernel[i].data = self.r_cholesk_kernel[i].data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            
            self.s_mean[i].data.normal_().to(self.weight.device)
            self.r_mean[i].data.normal_() .to(self.weight.device)
            self.s_mean[i].data =  self.s_mean[i].data.normal_() * 0.01
            self.r_mean[i].data =  self.r_mean[i].data.normal_() * 0.01
        #self.log_var.data.normal_()#.fill_(0.5)
        #self.log_var.data.fill_(0.0)
		
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        
        if var_reducer == 'Flipout':
            #eps_weight_s   = self.eps_weight_s.data.normal_()
            #sigma_weight_s = torch.exp(self.s_log_var[self.component_counter])
            #delta_weight_s = self.s_mean[self.component_counter]  + eps_weight_s*sigma_weight_s

            eps_weight_s   = self.eps_weight_s.data.normal_()
            eps_weight_pad = pad_circular_nd(eps_weight_s[None,:], self.kernel_size-1 , 1)
            delta_weight_s = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.s_cholesk_kernel[self.component_counter], dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight_s.shape[0]]              
            delta_weight_s = self.s_mean[self.component_counter] + delta_weight_s
            
            eps_weight_r   = self.eps_weight_r.data.normal_()
            eps_weight_pad = pad_circular_nd(eps_weight_r[None,:], self.kernel_size-1 , 1)
            delta_weight_r = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.r_cholesk_kernel[self.component_counter], dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight_r.shape[0]]              
            delta_weight_r = self.r_mean[self.component_counter] + delta_weight_r
            
            #eps_weight_r   = self.eps_weight_r.data.normal_()
            #sigma_weight_r = torch.exp(self.r_log_var[self.component_counter])
            #delta_weight_r = self.r_mean[self.component_counter] + eps_weight_r*sigma_weight_r
            
                        
            #print(self.weight.shape)
            #print((delta_weight_s @ delta_weight_r).reshape(self.in_features, self.out_features).T.shape)
            #print(  (delta_weight_s[:,None]  @ delta_weight_r[None,:]).shape )
            delta_weight = self.weight * (delta_weight_s[:,None]  @ delta_weight_r[None,:]).T
            
            outputs = F.linear(x, self.weight, self.bias)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            x_tmp = x * sign_input


            perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs

            self.component_counter = (self.component_counter + 1) % self.components
            return out
            
            
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))	SUM od components	
        kld = 0.0 # s_log_var[i]
        for i in range(self.components):
            
            paded_kernel = F.pad(self.r_cholesk_kernel[i], (0, self.r_mean[i].nelement()- self.kernel_size), 'constant')
            eigs = (torch.fft.fft(paded_kernel)).real**2
            kld += 0.5*(-self.r_mean[i].nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                        torch.sum(self.r_mean[i]*self.r_mean[i]) )
            
            paded_kernel = F.pad(self.s_cholesk_kernel[i], (0, self.s_mean[i].nelement()- self.kernel_size), 'constant')
            eigs = (torch.fft.fft(paded_kernel)).real**2
            kld += 0.5*(-self.s_mean[i].nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                        torch.sum(self.s_mean[i]*self.s_mean[i]) )

        return kld

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	 
    
class Conv2dRank1Circulant(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                circ_kernel_size,
                 init_scheme = 'Xavier',
                components = 3, 
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = False):
        
        super(Conv2dRank1Circulant, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)

        self.components = components
        self.component_counter = 0   
        self.circ_kernel_size = circ_kernel_size
        
        if circ_kernel_size > min(out_channels , in_channels):
            #raise Exception("Kernel size must be less or equal to the number of weights")
            print("The kernel size extends the weight number o elements, automatically set to " + str( min(out_channels , in_channels) ))
            self.circ_kernel_size = min(out_channels , in_channels) - 2
        else:
            self.circ_kernel_size = circ_kernel_size 
                
        
        self.s_mean    = nn.ParameterList( Parameter(torch.Tensor(in_channels)) for _ in range(components))
        self.s_cholesk_kernel =  nn.ParameterList( Parameter(torch.Tensor(self.circ_kernel_size)) for _ in range(components)) 
        
        self.r_mean    = nn.ParameterList( Parameter(torch.Tensor(out_channels)) for _ in range(components))
        self.r_cholesk_kernel =  nn.ParameterList( Parameter(torch.Tensor(self.circ_kernel_size)) for _ in range(components))
        
        self.register_buffer('eps_weight_s', torch.Tensor(in_channels), persistent=False) 
        self.register_buffer('eps_weight_r', torch.Tensor(out_channels), persistent=False)  
        
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            self.bias.data.zero_()  
        
        for i in range(self.components):
            self.s_cholesk_kernel[i].data.fill_(-5.0).to(self.weight.device)
            self.r_cholesk_kernel[i].data.fill_(-5.0).to(self.weight.device)
            
            self.s_cholesk_kernel[i].data = self.s_cholesk_kernel[i].data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            self.r_cholesk_kernel[i].data = self.r_cholesk_kernel[i].data.fill_(0.01) * np.sqrt( 2/(  self.weight.nelement()  ))  #XavierFIXED
            
            self.s_mean[i].data.normal_().to(self.weight.device)
            self.r_mean[i].data.normal_() .to(self.weight.device)
            self.s_mean[i].data =  self.s_mean[i].data.normal_() * 0.01
            self.r_mean[i].data =  self.r_mean[i].data.normal_() * 0.01   


    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))

        if var_reducer == 'Flipout':
            outputs = F.conv2d(x,
                               weight=self.weight,
                               bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            #sigma_weight = torch.exp(self.log_var)
            #eps_kernel = self.eps_kernel.data.normal_()
            #delta_kernel = (sigma_weight * eps_kernel)

            eps_weight_s   = self.eps_weight_s.data.normal_()
            eps_weight_pad = pad_circular_nd(eps_weight_s[None,:], self.circ_kernel_size-1 , 1)
            delta_weight_s = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.s_cholesk_kernel[self.component_counter], dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight_s.shape[0]]              
            delta_weight_s = self.s_mean[self.component_counter] + delta_weight_s
            
            eps_weight_r   = self.eps_weight_r.data.normal_()
            eps_weight_pad = pad_circular_nd(eps_weight_r[None,:], self.circ_kernel_size-1 , 1)
            delta_weight_r = torch.nn.functional.conv1d(eps_weight_pad[None,:] ,
                                        torch.flip( self.r_cholesk_kernel[self.component_counter], dims=(0,))[None,None,:], 
                                        padding=0)[0,0,0:eps_weight_r.shape[0]]              
            delta_weight_r = self.r_mean[self.component_counter] + delta_weight_r
            
            
            
            
            #delta_weight = self.weight * (delta_weight_s[:,None]  @ delta_weight_r[None,:]).T
            delta_kernel = self.weight  * (delta_weight_r[:,None]@delta_weight_s[None,:])[:,:,None,None]
            
            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.bias,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            
            self.component_counter = (self.component_counter + 1) % self.components
            return out    
                        
        else:
            raise Exception("Variance reducer must be either 'LRT' or 'Flipout' ") 

    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))	SUM od components	
        kld = 0.0 # s_log_var[i]
        for i in range(self.components):
            
            paded_kernel = F.pad(self.r_cholesk_kernel[i], (0, self.r_mean[i].nelement()- self.circ_kernel_size), 'constant')
            eigs = (torch.fft.fft(paded_kernel)).real**2
            kld += 0.5*(-self.r_mean[i].nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                        torch.sum(self.r_mean[i]*self.r_mean[i]) )
            
            paded_kernel = F.pad(self.s_cholesk_kernel[i], (0, self.s_mean[i].nelement()- self.circ_kernel_size), 'constant')
            eigs = (torch.fft.fft(paded_kernel)).real**2
            kld += 0.5*(-self.s_mean[i].nelement() - torch.sum(torch.log(eigs+0.0000001)) + torch.sum(eigs) + 
                        torch.sum(self.s_mean[i]*self.s_mean[i]) )

        return kld   
                
        
class LinearRank1(nn.Module):

    def __init__(self, in_features, out_features,components, bias=True):
        super(LinearRank1, self).__init__()
        
        # Dimentionality Parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
 
        self.s_mean    = nn.ParameterList( Parameter(torch.Tensor(in_features)) for _ in range(components))
        self.s_log_var =nn.ParameterList(  Parameter(torch.Tensor(in_features)) for _ in range(components))
        self.r_mean    = nn.ParameterList( Parameter(torch.Tensor(out_features)) for _ in range(components))
        self.r_log_var = nn.ParameterList( Parameter(torch.Tensor(out_features)) for _ in range(components))
        self.register_buffer('eps_weight_s', torch.Tensor(in_features), persistent=False) 
        self.register_buffer('eps_weight_r', torch.Tensor(out_features), persistent=False)  
              
        # Use bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.components = components
        self.component_counter = 0        
        self.reset_parameters()


    def reset_parameters(self): #TODO
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.W.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        #print('Ok')		
        #torch.nn.init.xavier_normal_(self.W, gain=1.0)	
        for i in range(self.components):
            self.s_log_var[i].data.fill_(-5.0).to(self.weight.device)
            self.r_log_var[i].data.fill_(-5.0).to(self.weight.device)
            self.s_mean[i].data.normal_().to(self.weight.device)
            self.r_mean[i].data.normal_() .to(self.weight.device)
            self.s_mean[i].data =  self.s_mean[i].data.normal_() * 0.01
            self.r_mean[i].data =  self.r_mean[i].data.normal_() * 0.01
        #self.log_var.data.normal_()#.fill_(0.5)
        #self.log_var.data.fill_(0.0)
		
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))
        if var_reducer == 'LRT':
            lrt_mean = F.linear(x, self.weight, self.bias)
            if self.bias is not None:
                lrt_mean = lrt_mean + self.bias

            sigma2 = torch.exp(self.log_var)
            lrt_std = torch.sqrt(1e-16 + F.linear(x * x, sigma2))

            if self.training:
                eps = lrt_std.data.new(lrt_std.size()).normal_()
            else:
                eps = 0.0
            return lrt_mean + lrt_std * eps
        
        
        if var_reducer == 'Flipout':
            eps_weight_s   = self.eps_weight_s.data.normal_()
            sigma_weight_s = torch.exp(self.s_log_var[self.component_counter])
            delta_weight_s = self.s_mean[self.component_counter]  + eps_weight_s*sigma_weight_s

            eps_weight_r   = self.eps_weight_r.data.normal_()
            sigma_weight_r = torch.exp(self.r_log_var[self.component_counter])
            delta_weight_r = self.r_mean[self.component_counter] + eps_weight_r*sigma_weight_r
            
                        
            #print(self.weight.shape)
            #print((delta_weight_s @ delta_weight_r).reshape(self.in_features, self.out_features).T.shape)
            #print(  (delta_weight_s[:,None]  @ delta_weight_r[None,:]).shape )
            delta_weight = self.weight * (delta_weight_s[:,None]  @ delta_weight_r[None,:]).T
            
            outputs = F.linear(x, self.weight, self.bias)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
            x_tmp = x * sign_input


            perturbed_outputs_tmp = F.linear(x_tmp, delta_weight, self.bias)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs

            self.component_counter = (self.component_counter + 1) % self.components
            return out
            
            
    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))	SUM od components	
        kld = 0.0 # s_log_var[i]
        for i in range(self.components):
            kld += 0.5*(-self.r_mean[i].nelement() - torch.sum(self.r_log_var[i]) + torch.sum(torch.exp(self.r_log_var[i])) + torch.sum(self.r_mean[i]*self.r_mean[i]))
            kld += 0.5*(-self.s_mean[i].nelement() - torch.sum(self.s_log_var[i]) + torch.sum(torch.exp(self.s_log_var[i])) + torch.sum(self.r_mean[i]*self.r_mean[i]))
        return kld

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'	 
    
    
    
class Conv2dRank1(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding = 0,
                dilation = 1,
                groups = 1,
                components = 3,
                bias = False):
        
        super(Conv2dRank1, self).__init__()
        
        # Dimentionality Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)

        self.components = components
        self.component_counter = 0   
         
        self.s_mean    = nn.ParameterList( Parameter(torch.Tensor(in_channels)) for _ in range(components))
        self.s_log_var =nn.ParameterList(  Parameter(torch.Tensor(in_channels)) for _ in range(components))
        self.r_mean    = nn.ParameterList( Parameter(torch.Tensor(out_channels)) for _ in range(components))
        self.r_log_var = nn.ParameterList( Parameter(torch.Tensor(out_channels)) for _ in range(components))
        self.register_buffer('eps_weight_s', torch.Tensor(in_channels), persistent=False) 
        self.register_buffer('eps_weight_r', torch.Tensor(out_channels), persistent=False)  
        
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            self.bias.data.zero_()  
        
        for i in range(self.components):
            self.s_log_var[i].data.fill_(-5.0).to(self.weight.device)
            self.r_log_var[i].data.fill_(-5.0).to(self.weight.device)
            self.s_mean[i].data.normal_().to(self.weight.device)
            self.r_mean[i].data.normal_() .to(self.weight.device)
            self.s_mean[i].data =  self.s_mean[i].data.normal_() * 0.01
            self.r_mean[i].data =  self.r_mean[i].data.normal_() * 0.01    


    def forward(self, x, var_reducer = "Flipout"):
        #print(torch.exp(self.log_var))

        if var_reducer == 'Flipout':
            outputs = F.conv2d(x,
                               weight=self.weight,
                               bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            #sigma_weight = torch.exp(self.log_var)
            #eps_kernel = self.eps_kernel.data.normal_()
            #delta_kernel = (sigma_weight * eps_kernel)

            eps_weight_s   = self.eps_weight_s.data.normal_()
            sigma_weight_s = torch.exp(self.s_log_var[self.component_counter])
            delta_weight_s = self.s_mean[self.component_counter]  + eps_weight_s*sigma_weight_s

            eps_weight_r   = self.eps_weight_r.data.normal_()
            sigma_weight_r = torch.exp(self.r_log_var[self.component_counter])
            delta_weight_r = self.r_mean[self.component_counter] + eps_weight_r*sigma_weight_r
            
            #delta_weight = self.weight * (delta_weight_s[:,None]  @ delta_weight_r[None,:]).T
            delta_kernel = self.weight  * (delta_weight_r[:,None]@delta_weight_s[None,:])[:,:,None,None]
            
            # perturbed feedforward
            x_tmp = x * sign_input
            perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                         weight=delta_kernel,
                                         bias=self.bias,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups)
            perturbed_outputs = perturbed_outputs_tmp * sign_output
            out = outputs + perturbed_outputs
            
            self.component_counter = (self.component_counter + 1) % self.components
            return out    
                        
        else:
            raise Exception("Variance reducer must be either 'LRT' or 'Flipout' ") 

    def kl_reg(self):
        # KL(q||p), where P = N(0,I)
        #print('lo det ', torch.sum(self.log_var))	SUM od components	
        kld = 0.0 # s_log_var[i]
        for i in range(self.components):
            kld += 0.5*(-self.r_mean[i].nelement() - torch.sum(self.r_log_var[i]) + torch.sum(torch.exp(self.r_log_var[i])) + torch.sum(self.r_mean[i]*self.r_mean[i]))
            kld += 0.5*(-self.s_mean[i].nelement() - torch.sum(self.s_log_var[i]) + torch.sum(torch.exp(self.s_log_var[i])) + torch.sum(self.r_mean[i]*self.r_mean[i]))
        return kld     
        
        
        
        
        