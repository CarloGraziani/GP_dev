import torch
from torch import Tensor
import gpytorch
import linear_operator
from experimental_uncertainty_kernel import ExperimentalUncertaintyKernel, fix_lengthscale
from scipy.stats import chi2
import sys
import time

torch.set_default_dtype(torch.float64)
sys.path.insert(0,"..")
sys.path.insert(0,".")

# Define parameters to be initialized by jobctl

data_dir = "../../Data/"
data_file = "roi_all_comps.pt"
test_samp = False
init_params = False
training_iterations = False
save_state_cadence = False
time_limit = False
max_cg = False
lr = False
max_prec_size = False
prec_tolerance = False
octl = False


# jobctl is a dict containing control parameters and the GPyTorch model class EPUModel 
from train_in import jobctl, EPUModel
globals().update(jobctl)


#### Load and prep the data
dt = torch.load(data_dir+data_file)
k = list(dt.keys)
expts= dt[k[0]] ; spec = dt[k[1]]

block_shape = spec.shape ; dim = expts.shape[-1]
nspecbin = block_shape[1] ; nspec = block_shape[0]

# a test sample every test_samp bins
ind = torch.arange(nspecbin)
train_ind = ind[(ind % test_samp).to(int) != 0]
test_ind = ind[(ind % test_samp).to(int) == 0]

# Training data
tx = expts[:,train_ind,:]
train_x = tx.reshape((-1, dim))
ty = spec[:,train_ind]
train_y = ty.flatten()

noise = train_y

dof = len(train_y)

# Test and validation data
test_x = expts[:,test_ind,:]
test_y = spec[:,test_ind]

val_x = test_x[0] ; val_y = test_y[0] ; val_noise = val_y
test_x = test_x[1:] ; test_y = test_y[1:] ; test_noise = test_y

# Put everything on GPU
train_x = train_x.cuda()
train_y = train_y.cuda()
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
        pass


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
            mn = model.mean_module.constant.item()

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
                loss2 = inv_quad_t + logdet_t

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
                     f"\n  Mean = {mn:16.9E};"

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