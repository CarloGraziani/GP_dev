import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from experimental_uncertainty_kernel import ExperimentalUncertaintyKernel, fix_lengthscale
import sys
sys.path.insert(0,"..")

jobctl = {
    'thin_data': 8, # Factor by which data is thinned out
    'test_samp': 1.8, # a test sample every test_samp bins
    'zscore_input': True,
    'zscore_output': True,
    'mask': [71], # List of experiments to leave out of the analysis
#
#Initialization
    'init_params': {
    # 'mean_module.constant': 0.0E+00,
    'covar_module.kernels.0.outputscale': 0.6E+00,
    'covar_module.kernels.0.base_kernel.lengthscale': Tensor([0.1, 5.6, 2.9]).cuda(),
    'covar_module.kernels.1.os_0':-3.2E+00,
    'covar_module.kernels.1.os_v': Tensor([0.21, -0.16]).cuda(),
    'covar_module.kernels.1.ls_0': -2.3E+00,
    # 'covar_module.kernels.1.ls_v': Tensor([-0.55, -0.01]).cuda(),
    'covar_module.kernels.2.os_0':-3.0E+00,
    # 'covar_module.kernels.1.os_v': Tensor([0.21, -0.16]).cuda(),
    'covar_module.kernels.2.ls_0': -2.3E+00,
    # 'covar_module.kernels.1.ls_v': Tensor([-0.55, -0.01]).cuda(),
    }, 
#
# training parameters
    'training_iterations': 500,
    'save_state_cadence': 10,
    'time_limit': 59 * 60,
    'max_cg': 10000,
    'lr': 0.02,
    'max_prec_size': 60,
    'prec_tolerance': 1.0E-04,

# Output control
    'octl': [
        #    ("Mean", "model.mean_module.constant.item()", "16.9E"),
           ("Chem_os_1", "model.covar_module.kernels[0].outputscale.item()", "16.9E"),
           ("Chem_ls_1", "model.covar_module.kernels[0].base_kernel.lengthscale[0].tolist()", ""),
           ("os_0", "model.covar_module.kernels[1].os_0.item()", "16.9E"),
           ("os_v", "model.covar_module.kernels[1].os_v.tolist()", ""),
           ("ls_0", "model.covar_module.kernels[1].ls_0.item()", "16.9E"),
        #    ("ls_v", "model.covar_module.kernels[1].ls_v.tolist()", ""),
           ("os_1", "model.covar_module.kernels[2].os_0.item()", "16.9E"),
           ("ls_1", "model.covar_module.kernels[2].ls_0.item()", "16.9E"),
           ],
}

#### Create the GP Model
class EPUModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nspec):
        super(EPUModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        dim = train_x.shape[-1]

        # Chemistry covariance kernel
        covar_chem_g1 = ScaleKernel(MaternKernel(ard_num_dims=dim, nu=1.5))

        # Experimental procedure uncertainty kernel
        SMaternKernel = fix_lengthscale(MaternKernel) #lengthscale and outputscale handled by ExperimentalUncertaintyKernel
        base_kernel1 = SMaternKernel(nu=1.5)
        base_kernel1.lengthscale = base_kernel1.lengthscale.cuda()
        data_size = len(train_y) // nspec
        exp_par_ind = torch.arange(start=1, end=train_x.shape[-1])
        covar_eu1 = ExperimentalUncertaintyKernel(base_kernel1, train_x, data_size, 
                                                 exp_par_ind, outputscale_fn=True, 
                                                 lengthscale_fn=False)
        base_kernel2 = SMaternKernel(nu=1.5)
        base_kernel2.lengthscale = base_kernel2.lengthscale.cuda()
        data_size = len(train_y) // nspec
        exp_par_ind = torch.arange(start=1, end=train_x.shape[-1])
        covar_eu2 = ExperimentalUncertaintyKernel(base_kernel2, train_x, data_size, 
                                                 exp_par_ind, outputscale_fn=False, 
                                                 lengthscale_fn=False)
        
        # self.covar_module = covar_chem + covar_eu1 + covar_eu2
        self.covar_module = covar_chem_g1 + covar_eu1 + covar_eu2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)