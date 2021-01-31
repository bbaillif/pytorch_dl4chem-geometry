import torch

# maybe mask could be taken out of the function

class KLDLoss(object) :
    """Used to compute the Kullback-Leibler divergence (KLD) between 2 distributions
    One distribution can be given as input, computing KLD between input and normal distrution"""
# KL(p,q)= log(σ2/σ1) + ((σ1²+(μ1−μ2)²)/ 2σ2²) − (1/2)
        
    def loss(self, mask, mu0, lsgm0, mu1=None, lsgm1=None) :
        if mu1 is None and lsgm1 is None :
            return self._KLD_zero(mu0, lsgm0, mask)
        else :
            return self._KLD(mu0, lsgm0, mu1, lsgm1, mask)
        
        
    def _KLD(self, mask, mu0, lsgm0, mu1, lsgm1):
        """
        Args :
            mu0 : Tensor(batch_size, n_max, dim_h) : means of the first multivariate
            lsgm0 : Tensor(batch_size, n_max, dim_h) : log of the variances of the first multivariate 
            mu1 : Tensor(batch_size, n_max, dim_h) : means of the second multivariate
            lsgm1 : Tensor(batch_size, n_max, dim_h) : log of the variances of the second multivariate
            mask : Tensor(batch_size, n_max, 1) : existing atom masking (1 = atom exist ; 0 = padding)

        Returns :
            kld : Tensor(batch_size, n_max, dim_h) : KL divergence between the 2 distributions
        """


        var0 = torch.exp(lsgm0) + 1e-5
        var1 = torch.exp(lsgm1) + 1e-5
        a = torch.div(var0,var1)
        b = torch.div(torch.square(torch.sub(mu1, mu0)), var1)
        c = torch.log(torch.div(var1,var0))

        kld = 0.5 * torch.sum(a + b - 1 + c, 2, keepdim=True) * mask

        return kld


    def _KLD_zero(self, mask, mu0, lsgm0):
        """
        Args :
            mu0 : Tensor(batch_size, n_max, dim_h) : means of the input multivariate
            lsgm0 : Tensor(batch_size, n_max, dim_h) : log of the variances of the input multivariate 
            mask : Tensor(batch_size, n_max, 1) : existing atom masking (1 = atom exist ; 0 = padding)

        Returns :
            kld : Tensor(batch_size, n_max, dim_h) : KLD between the input distribution and a normal gaussian
        """


        a = torch.exp(lsgm0) + torch.square(mu0)
        b = 1 + lsgm0

        kld = 0.5 * torch.sum(a - b, 2, keepdim=True) * mask

        return kld