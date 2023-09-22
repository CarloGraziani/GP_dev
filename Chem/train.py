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

data_dir = "../../Data/"
data_file = "roi_all_comps.pt"
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
roi_selection = None
zscore_input = False
zscore_output = False
zscore_per_expt = False
scale_wns = None
scale_out = None
report_full_validation = False

# jobctl is a dict containing control parameters and the GPyTorch model class EPUModel 
from train_in import jobctl, EPUModel
globals().update(jobctl)


#### Load and prep the data
dt = torch.load(data_dir+data_file)
k = list(dt.keys)
expts= dt[k[0]] ; spec = dt[k[1]]

roi_bds = Tensor([[889.89, 1090.0], [1270.0, 1360.0], [1379.99, 1560.0], [1589.95, 1770.0]])

if mask is not None:
    nexp = spec.shape[0]
    ind = [i for i in range(nexp) if not i in mask]
    expts = expts[ind, :, :]
    spec = spec[ind, :]

if roi_selection is not None:
    mask = torch.zeros_like(spec[0,:]).to(bool)
    for roi in roi_selection:
        m = torch.logical_and(expts[0,:,0] > roi_bds[roi, 0], 
                              expts[0,:,0] < roi_bds[roi, 1])
        mask = torch.logical_or(mask, m)
    spec = spec[:, mask]
    expts = expts[:, mask, :]

if thin_data is not None:
    ind = torch.arange(spec.shape[1])
    ind =ind[(ind % thin_data).to(int) == 0]
    expts = expts[:, ind, :]
    spec = spec[:, ind]

in_mean = 0.0 ; in_std = 1.0
if zscore_input:
    if zscore_per_expt:
        # Use wavenumber range of 1st experiment to zscore the wavenumbers only
        in_mean = expts[0, :, 0].mean()
        in_std = expts[0, :, 0].std()
        expts[:,:,0] = (expts[:,:,0] - in_mean) / in_std
    else:
        in_mean = expts.mean(dim=(0,1))
        in_std = expts.std(dim=(0,1))
        expts = (expts - in_mean) / in_std

out_mean = 0.0 ; out_std = 1.0
if zscore_output:
    if zscore_per_expt:
        out_mean = spec.mean(dim=-1, keepdims=True)
        out_std = spec.std(dim=-1, keepdims=True)
    else:
        out_mean = spec.mean()
        out_std = spec.std()
    noise = spec / out_std**2 
    spec = (spec - out_mean) / out_std

if scale_wns is not None:
    lengths = torch.exp( scale_wns[0] + expts[:,:,1:].matmul(scale_wns[1:]) )
    lengths = lengths / lengths[0,0]
    expts[...,0] = expts[...,0] * lengths

if scale_out is not None:
    oscls = torch.exp( scale_out[0] + expts[:,:,1:].matmul(scale_out[1:]) )
    oscls = oscls / oscls[0,0]
    spec = spec / oscls

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
noise = noise[:, train_ind]
noise = noise.flatten()

dof = len(train_y)
c2 = chi2(dof)

# Test and validation data
val_x = expts[:,test_ind,:]
val_y = spec[:,test_ind]
val_noise = noise[:, test_ind]

# val_x = test_x[0] ; val_y = test_y[0] ; val_noise = t_noise[0]
# test_x = test_x[1:] ; test_y = test_y[1:] ; test_noise = t_noise[1:]

# Put everything on GPU
train_x = train_x.cuda()
train_y = train_y.cuda()
noise = noise.cuda()
val_x = val_x.cuda()
val_y = val_y.cuda()
val_noise = val_noise.cuda()
dof_val = val_y.shape[-1]
c2_val = chi2(dof_val)

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
                inv_quad_t = inv_quad_t.cpu().detach()
                pvalue = c2.cdf(inv_quad_t)
                qvalue = c2.sf(inv_quad_t)
                loss2 = 0.5 * ( (inv_quad_t + logdet_t)/train_y.shape[-1] + log(2*pi) )

                model.eval() ; likelihood.eval()

                chi2_val = [] ; logpval_val = [] ; logsfval_val = []
                with gpytorch.settings.fast_pred_var():
                    for l in range(nspec):
                        mvn = model(val_x[l,...])
                        lik = likelihood(mvn, noise=val_noise[l,:])
                        mean, covar = lik.loc, lik.lazy_covariance_matrix
                        diff = (val_y[l,:] - mean).unsqueeze(-1)
                        covar = covar.evaluate_kernel()
                        inv_quad_vf = covar.inv_quad(inv_quad_rhs=diff).cpu().detach()
                        pvalue_val = c2_val.logcdf(inv_quad_vf)
                        qvalue_val = c2_val.logsf(inv_quad_vf)
                        chi2_val.append(inv_quad_vf)
                        logpval_val.append(pvalue_val)
                        logsfval_val.append(qvalue_val)

                chi2_val_mn = Tensor(chi2_val).mean()
                chi2_val_md = Tensor(chi2_val).quantile(q=0.5)
                logpval_val_mn = Tensor(logpval_val).mean()
                logpval_val_md = Tensor(logpval_val).quantile(q=0.5)
                logsfval_val_mn = Tensor(logsfval_val).mean()
                logsfval_val_md = Tensor(logsfval_val).quantile(q=0.5)

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
                f"\n  Training Chi-squared = {inv_quad_t:16.9E}, DOF = {dof:d},"+\
                f"\n                     P = {pvalue:10.3E}, 1-P = {qvalue:10.3E}"+\
                f"\n  Training Log Det = {logdet_t:16.9E}"+\
                f"\n  Mean Val. Chi-squared = {chi2_val_mn:16.9E}, DOF = {dof_val:d},"+\
                f"\n            Mean log(P) = {logpval_val_mn:10.3E}, Mean log(1-P) = {logsfval_val_mn:10.3E}"+\
                f"\n  Median Val. Chi-squared = {chi2_val_md:16.9E}, DOF = {dof_val:d},"+\
                f"\n            Median log(P) = {logpval_val_mn:10.3E}, Median log(1-P) = {logsfval_val_mn:10.3E}"
            if report_full_validation:
                outstr += \
                "\n Val Chi_Squared: [" + ("{:16.9E} ," * (nspec-1)).format(*chi2_val[:-1]) + "{:16.9E}]".format(chi2_val[-1]) +\
                "\n Val Log P: [" + ("{:16.9E} ," * (nspec-1)).format(*logpval_val[:-1]) + "{:16.9E}]".format(logpval_val[-1]) +\
                "\n Val Log(1-P): [" + ("{:16.9E} ," * (nspec-1)).format(*logsfval_val[:-1]) + "{:16.9E}]".format(logsfval_val[-1])

            outstr += \
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