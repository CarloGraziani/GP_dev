import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from experimental_uncertainty_kernel import ExperimentalUncertaintyKernel, fix_lengthscale
import sys
sys.path.insert(0,"..")

jobctl = {
    'test_samp': 1.8, # a test sample every test_samp bins
#
#Initialization
    'init_params': {
    'covar_module.kernels.0.outputscale': 2.47121E+08/2.0,
    'covar_module.kernels.0.base_kernel.lengthscale': Tensor([7.5, 11.4, 11.4]).cuda(),
    'covar_module.kernels.1.outputscale': 2.47121E+08/2.0,
    'covar_module.kernels.1.base_kernel.lengthscale': Tensor([7.5, 2.0, 2.0]).cuda(),
    'covar_module.kernels.2.os_0': 1.57809E+01,
    'covar_module.kernels.2.os_v': Tensor([0.0, 0.0]).cuda(),
    'covar_module.kernels.2.ls_0': 2.2E+00,
    'covar_module.kernels.2.ls_v': Tensor([0.0, 0.0]).cuda(),
    }, 
#
# training parameters
    'training_iterations': 400,
    'save_state_cadence': 5,
    'time_limit': 59 * 60,
    'max_cg': 10000,
    'lr': 0.005,
    'max_prec_size': 20,
    'prec_tolerance': 1.0E-04,

# Output control
    'octl': [("Chem_os_1", "model.covar_module.kernels[0].outputscale.item()", "16.9E"),
           ("Chem_ls_1", "model.covar_module.kernels[0].base_kernel.lengthscale[0].tolist()", ""),
           ("Chem_os_2", "model.covar_module.kernels[1].outputscale.item()", "16.9E"),
           ("Chem_ls_2", "model.covar_module.kernels[1].base_kernel.lengthscale[0].tolist()", ""),
           ("os_0", "model.covar_module.kernels[2].os_0.item()", "16.9E"),
           ("os_v", "model.covar_module.kernels[2].os_v.tolist()", ""),
           ("ls_0", "model.covar_module.kernels[2].ls_0.item()", "16.9E"),
           ("ls_v", "model.covar_module.kernels[2].ls_v.tolist()", ""),
           ],
}

#### Create the GP Model
class EPUModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nspec):
        super(EPUModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        dim = train_x.shape[-1]

        # Chemistry covariance kernel
        covar_chem_g1 = ScaleKernel(MaternKernel(ard_num_dims=dim, nu=1.5))
        covar_chem_g2 = ScaleKernel(RBFKernel(ard_num_dims=dim))

        # Experimental procedure uncertainty kernel
        SMaternKernel = fix_lengthscale(MaternKernel) #lengthscale and outputscale handled by ExperimentalUncertaintyKernel
        base_kernel1 = SMaternKernel(nu=1.5)
        base_kernel1.lengthscale = base_kernel1.lengthscale.cuda()
        data_size = len(train_y) // nspec
        exp_par_ind = torch.arange(start=1, end=train_x.shape[-1])
        covar_eu1 = ExperimentalUncertaintyKernel(base_kernel1, train_x, data_size, 
                                                 exp_par_ind, outputscale_fn=True, 
                                                 lengthscale_fn=True)
        # base_kernel2 = SMaternKernel(nu=2.5)
        # base_kernel2.lengthscale = base_kernel2.lengthscale.cuda()
        # data_size = len(train_y) // nspec
        # exp_par_ind = torch.arange(start=1, end=train_x.shape[-1])
        # covar_eu2 = ExperimentalUncertaintyKernel(base_kernel2, train_x, data_size, 
        #                                          exp_par_ind, outputscale_fn=True, 
        #                                          lengthscale_fn=True)
        
        # self.covar_module = covar_chem + covar_eu1 + covar_eu2
        self.covar_module = covar_chem_g1 + covar_chem_g2 + covar_eu1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)