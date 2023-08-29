import torch
from torch import Tensor
import gpytorch
import linear_operator
from scipy.stats import chi2
from math import log, pi
import sys
import time

sys.path.insert(0,"..")
sys.path.insert(0,".")
from experimental_uncertainty_kernel import ExperimentalUncertaintyKernel, fix_lengthscale

torch.set_default_dtype(torch.float64)

# Define parameters to be initialized by jobctl
thin_data = None
test_samp = None
init_params = None
training_iterations = None
save_state_cadence = None
time_limit = None
max_cg = None
lr = None
max_prec_size = None
prec_tolerance = None
octl = None
mask=None
zscore_input = False
zscore_output = False

# jobctl is a dict containing control parameters and the GPyTorch model class EPUModel 
from train_in import jobctl, EPUModel
globals().update(jobctl)


#### Load and prep the data
data_dir = "../../Data/"
dt = torch.load(data_dir+"roi_all_comps.pt")
expts_roi = dt["expts_roi"] ; spec_roi = dt["spec_roi"]

if mask is not None:
    nexp = spec_roi.shape[0]
    ind = [i for i in range(nexp) if not i in mask]
    expts_roi = expts_roi[ind, :, :]
    spec_roi = spec_roi[ind, :]


if thin_data is not None:
    ind = torch.arange(spec_roi.shape[1])
    ind =ind[(ind % thin_data).to(int) == 0]
    expts_roi = expts_roi[:, ind, :]
    spec_roi = spec_roi[:, ind]

in_mean = 0.0 ; in_std = 1.0
if zscore_input:
    in_mean = expts_roi.mean(dim=(0,1))
    in_std = expts_roi.std(dim=(0,1))
    expts_roi = (expts_roi - in_mean) / in_std

out_mean = 0.0 ; out_std = 1.0
if zscore_output:
    out_mean = spec_roi.mean()
    out_std = spec_roi.std()
    noise_roi = spec_roi / out_std**2 
    spec_roi = (spec_roi - out_mean) / out_std

block_shape = spec_roi.shape ; dim = expts_roi.shape[-1]
nspecbin = block_shape[1] ; nspec = block_shape[0]

# a test sample every test_samp bins
ind = torch.arange(nspecbin)
train_ind = ind[(ind % test_samp).to(int) != 0]
test_ind = ind[(ind % test_samp).to(int) == 0]

# Training data
tx = expts_roi[:,train_ind,:]
train_x = tx.reshape((-1, dim))
ty = spec_roi[:,train_ind]
train_y = ty.flatten()
noise = noise_roi[:, train_ind]
noise = noise.flatten()

dof = len(train_y)

# Test and validation data
test_x = expts_roi[:,test_ind,:]
test_y = spec_roi[:,test_ind]
t_noise = noise_roi[:, test_ind]

val_x = test_x[0] ; val_y = test_y[0] ; val_noise = t_noise[0]
test_x = test_x[1:] ; test_y = test_y[1:] ; test_noise = t_noise[1:]

# Put everything on GPU
train_x = train_x.cuda()
train_y = train_y.cuda()
noise = noise.cuda()
val_x = val_x.cuda()
val_y = val_y.cuda()
val_noise = val_noise.cuda()
dof_val = len(val_y)

## Set up the model    
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
model = EPUModel(train_x, train_y, likelihood, nspec)

# Put everything on GPU
model = model.cuda()
likelihood = likelihood.cuda()

# for pn, _ in model.named_parameters():
#     print(pn)

# sys.exit()

# Initialization of parameters
if hasattr(model.mean_module, "constant") and not init_params.get("mean_module.constant"):
    mn_init = train_y.mean()
    init_params["mean_module.constant"]=mn_init
model.initialize(**init_params)


#### Training


# Checkpoint?
curr_iter = 0
if len(sys.argv) > 1:
    try:
        state_dict = torch.load(sys.argv[1])
        model.load_state_dict(state_dict)
        curr_iter = int(sys.argv[1][16:21]) + 1
    except FileNotFoundError:
        print(f"File {sys.argv[1]} not found!")
        raise


model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

tick_start = time.perf_counter()
tick = tick_start
with open("model_output.txt", "a") as ofd:

    for iter in range(curr_iter, curr_iter+training_iterations):

        with linear_operator.settings.max_cg_iterations(max_cg),\
             linear_operator.settings.max_preconditioner_size(max_prec_size),\
             linear_operator.settings.preconditioner_tolerance(prec_tolerance):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            # mn = model.mean_module.constant.item()

            vals = [] ; fmts = [] ; names = []
            val=0
            for output in octl:
                names.append(output[0])
                exec("val = "+output[1])
                vals.append(val)
                fmts.append(output[2])

            with torch.no_grad():
                mvn = model(train_x)
                lik = likelihood(mvn)
                mean, covar = lik.loc, lik.lazy_covariance_matrix
                diff = (train_y - mean).unsqueeze(-1)
                covar = covar.evaluate_kernel()
                inv_quad_t, logdet_t = covar.inv_quad_logdet(inv_quad_rhs=diff, logdet=True)
                loss2 = 0.5 * ( (inv_quad_t + logdet_t)/train_y.shape[-1] + log(2*pi) )

                model.eval() ; likelihood.eval()

                with gpytorch.settings.fast_pred_var():
                    mvn = model(val_x)
                    lik = likelihood(mvn, noise=val_noise)
                    mean, covar = lik.loc, lik.lazy_covariance_matrix
                    diff = (val_y - mean).unsqueeze(-1)
                    covar = covar.evaluate_kernel()
                    inv_quad_vf = covar.inv_quad(inv_quad_rhs=diff)

                model.train() ; likelihood.train()

            tock = time.perf_counter()
            dt = tock - tick
            t_elapsed = tick - tick_start
            tick = tock

            outstr = f"{iter}: Loss = {loss.item():10.3E};"+ \
                     f"\n  Evaluated loss = {loss2:10.3E};"

            fstr = ""
            for i in range(len(names)):
                fstr += "\n  {1} = {{{0}:{2}}}".format(i, names[i], fmts[i])
            outstr += fstr.format(*vals)
            outstr += \
                f"\n  Training Chi-squared = {inv_quad_t:16.9E}, DOF = {dof:d}"+\
                f"\n  Training Log Det = {logdet_t:16.9E}"+\
                f"\n  Fast Validation Chi-squared = {inv_quad_vf:16.9E}, DOF = {dof_val:d}"+\
                f"\n  Elapsed Time in This Iteration = {dt:.2f}"+\
                "\n"
            ofd.write(outstr)
            ofd.flush()
            print(outstr)

            if (iter % save_state_cadence == 0
                or iter== curr_iter+training_iterations-1):
                filename = f"state_dict_save_{iter:05d}.pth"
                torch.save(model.state_dict(), filename)

            if t_elapsed > time_limit:
                ofd.write("Elapsed time exceeds time limit. Exiting training loop.")
                print("Elapsed time exceeds time limit. Exiting training loop.")
                filename = f"state_dict_save_{iter:05d}.pth"
                torch.save(model.state_dict(), filename)
                break