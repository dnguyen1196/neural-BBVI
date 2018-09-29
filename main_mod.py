from math import pi

import scipy
import autograd.numpy as ag_np
import autograd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import copy
import time
import sys
import os
from scipy.stats import norm
from scipy.interpolate import Rbf

np.random.seed(42)

# Store the variational parameters also in dictionary layers similar to a neural network
def make_nn_params_as_list_of_dicts(n_hiddens_per_layer_list=[5],\
        n_dims_input=1,\
        n_dims_output=1,\
        weight_fill_func=np.zeros,\
        bias_fill_func=np.zeros):
    nn_param_list = []
    n_hiddens_per_layer_list = [n_dims_input] + n_hiddens_per_layer_list + [n_dims_output]

    # Given full network size list is [a, b, c, d, e]
    # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
    for n_in, n_out in zip(n_hiddens_per_layer_list[:-1], n_hiddens_per_layer_list[1:]):
        nn_param_list.append(
            dict(
                w=weight_fill_func((n_in, n_out)),
                b=bias_fill_func((n_out,)),
            ))
    return nn_param_list


def predict_y_given_x_with_NN(x=None, nn_param_list=None, activation_func=np.tanh):
    """
    Predict y value given x value via feed-forward neural net
    Args
    ----
    x : array_like, n_examples x n_input_dims

    Returns
    -------
    y : array_like, n_examples
    """
    for layer_id, layer_dict in enumerate(nn_param_list):
        if layer_id == 0:
            if x.ndim > 1:
                in_arr = x
            else:
                if x.size == nn_param_list[0]['w'].shape[0]:
                    in_arr = x[ag_np.newaxis,:]
                else:
                    in_arr = x[:,ag_np.newaxis]
        else:
            in_arr = activation_func(out_arr)
        out_arr = ag_np.dot(in_arr, layer_dict['w']) + layer_dict['b']
    return ag_np.squeeze(out_arr)


def make_posterior_params_as_list_of_dicts(n_hiddens_per_layer_list=[5],\
        n_dims_input=1,\
        n_dims_output=1,\
        mean_weight_fill_func=np.zeros,\
        log_std_weight_full_func=np.zeros,
        mean_bias_fill_func=np.zeros,
        log_std_bias_fill_func=np.zeros):
    posterior_param_list = []
    n_hiddens_per_layer_list = [n_dims_input] + n_hiddens_per_layer_list + [n_dims_output]
    #print("hidden", n_hiddens_per_layer_list)
    # Given full network size list is [a, b, c, d, e]
    # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
    for n_in, n_out in zip(n_hiddens_per_layer_list[:-1], n_hiddens_per_layer_list[1:]):
        posterior_param_list.append(
            dict(
                mw=mean_weight_fill_func((n_in, n_out)),
                sw=log_std_weight_full_func((n_in, n_out)),
                mb=mean_bias_fill_func((n_out,)),
                sb=log_std_bias_fill_func((n_out,)),
            ))
    return posterior_param_list


def sample_neural_network_from_posterior(posterior_means, posterior_log_std, n_hiddens_per_layer_list, n_dims_input=1, n_dims_output=1):
    nn_param_list = []
    n_hiddens_per_layer_list = [n_dims_input] + n_hiddens_per_layer_list + [n_dims_output]

    # Given full network size list is [a, b, c, d, e]
    # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
    n_units_in = n_hiddens_per_layer_list[:-1]
    n_units_out = n_hiddens_per_layer_list[1:]
    for i in range(len(n_units_in)):
        n_in = n_units_in[i]
        n_out = n_units_out[i]

        mean_params = posterior_means[i]
        log_std_params = posterior_log_std[i]

        weight_mean = mean_params["w"]
        weight_log_std = log_std_params["w"]

        bias_mean = mean_params["b"]
        bias_log_std = log_std_params["b"]

        nn_param_list.append(
            dict(
                w=np.random.normal(weight_mean, np.exp(weight_log_std),size=(n_in, n_out)),
                b=np.random.normal(bias_mean, np.exp(bias_log_std), size=(n_out,)),
        ))
    return nn_param_list


def compute_log_likelihood(nn_params, x_train_N, y_train_N):
    sigma = 0.1
    yhat = predict_y_given_x_with_NN(x_train_N, nn_params, ag_np.tanh)
    #pdfs = scipy.stats.norm.logpdf(y_train_N, yhat, sigma)
    #log_pdfs = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * np.square(y_train_N - yhat)/sigma**2
    log_pdfs = -0.5 * np.square(y_train_N - yhat)/sigma**2
    return np.sum(log_pdfs)
    #for i in range(len(yhat)):
    #    logpdf += scipy.stats.norm.logpdf(y_train_N[i], yhat[i], sigma)
    #return logpdf

def compute_log_prior(nn_params):
    meanp = 0
    sigmap = 1
    logprior = 0.0
    for layer_id, layer in enumerate(nn_params):
        #logprior += np.log(1/(np.sqrt(2*pi*sigmap**2))) + -0.5 * np.sum(np.square(layer["w"]-meanp))/sigmap**2
        logprior += -0.5 * np.sum(np.square(layer["w"]-meanp))
        #logprior += np.log(1/(np.sqrt(2*pi*sigmap**2))) + -0.5 * np.sum(np.square(layer["b"]-meanp))/sigmap**2
        logprior += -0.5 * np.sum(np.square(layer["b"]-meanp))
    return logprior

def compute_log_q_posterior(nn_params, posterior_means, posterior_log_std):
    logposterior = 0.0
    for layer_id, layer in enumerate(nn_params):
        mean_params = posterior_means[layer_id]
        log_std_params  = posterior_log_std[layer_id]

        mw = mean_params["w"]
        sw = np.exp(log_std_params["w"])

        # Note that posterior params store the log of the standard deviation
        mb = mean_params["b"]
        sb = np.exp(log_std_params["b"])

        logposterior += -0.5 * np.sum(2*pi*np.square(sw)) + -0.5 * np.sum(np.square(layer["w"]-mw)/np.square(sw))
        logposterior += -0.5 * np.sum(2*pi*np.square(sb)) + -0.5 * np.sum(np.square(layer["b"]-mb)/np.square(sb))
    return logposterior

# Function to estimate VI loss function or minus ELBO
# Takes in parameter(mbar, sbar, mtilde, ms)
def estimate_ELBO(posterior_means, posterior_log_std, x_train_N, y_train_N, num_samples, nn_structure):
    """
    The ELBO is computed as

    L(m) = E_q(z|m) [log p(y|x, z) + log p(z) - log q(z|m)]
    """
    elbo_sampled_vals = [0 for _ in range(num_samples)]

    for num in range(num_samples):
        nn_param_sampled = sample_neural_network_from_posterior(posterior_means, posterior_log_std, nn_structure)
        log_ll = compute_log_likelihood(nn_param_sampled, x_train_N, y_train_N)
        log_prior = compute_log_prior(nn_param_sampled)
        log_q_posterior = compute_log_q_posterior(nn_param_sampled, posterior_means, posterior_log_std)

        elbo_sampled_vals[num] = log_ll + log_prior - log_q_posterior

    return np.average(elbo_sampled_vals)


def estimate_VI_loss(posterior_means, posterior_log_std, x_train_N, y_train_N, num_samples, nn_structure):
    return -estimate_ELBO(posterior_means, posterior_log_std, x_train_N, y_train_N, num_samples, nn_structure)


def compute_derivative_log_posterior(posterior_means, posterior_log_std, nn_param_sample):
    derivative_mean = copy.deepcopy(posterior_means)
    derivative_log_std = copy.deepcopy(posterior_log_std)

    for layer_id, layer in enumerate(posterior_means):
        w = nn_param_sample[layer_id]["w"]
        b = nn_param_sample[layer_id]["b"]

        mw = np.copy(derivative_mean[layer_id]["w"])
        sw = np.copy(derivative_log_std[layer_id]["w"])
        mb = np.copy(derivative_mean[layer_id]["b"])
        sb = np.copy(derivative_log_std[layer_id]["b"])

        assert(w.shape == mw.shape)
        assert(b.shape == mb.shape)
        assert(w.shape == sw.shape)
        assert(b.shape == sb.shape)

        #derivative_param[layer_id]["mw"] = (layer["mw"]-w)/np.exp(2*layer["sw"])
        derivative_mean[layer_id]["w"]  = (w - mw)/np.exp(2*sw)
        derivative_log_std[layer_id]["w"] = -2*sw + np.square(mw-w)/np.exp(2*sw)
        derivative_mean[layer_id]["b"] = (b - mb)/np.exp(2*sb)
        derivative_log_std[layer_id]["b"] = -2*sb + np.square(mb-b)/np.exp(2*sb)

    return derivative_mean, derivative_log_std


def add_two_list_dictionaries(l1, l2, alpha):
    # Perform l1 += alpha * l2
    # It must be that l1 and l2 have the same set of 
    # keys and structures
    for layer_id, layer in enumerate(l1):
        for key in layer.keys():
            layer[key] = np.add(layer[key], alpha * l2[layer_id][key])
    return l1


# Function to estimate the gradient of the loss funcion or minus derivative of ELBO
def compute_expected_gradient_ELBO(posterior_means, posterior_log_std, x_train_N, y_train_N, num_samples, nn_structure):
    """
    The gradient of the ELBO is given as
    dL/dm = E_q(z|m) [ dlog q(z|m)/dm (log p(y|x,z) + log p(z) - log q(z)) ]

    where the logLL, log prior and log posterior can be computed as usual
    The derivative of log posterior with respect to dm has closed form

    since q(z|m) is Gaussian and factorizes into individual gaussian on wi and bi

    dlog q/dmi = (wi- mi)/e^(2si)
    dlog q/dsi = (wi- mi)**2/e^(2si)
    """
    #expected_gradient_elbo = make_posterior_params_as_list_of_dicts(nn_structure)
    expected_gradient_mean = make_nn_params_as_list_of_dicts(nn_structure)
    expected_gradient_log_std = make_nn_params_as_list_of_dicts(nn_structure)

    #print("before", expected_gradient_elbo)
    for num in range(num_samples):
        nn_param_sample = sample_neural_network_from_posterior(posterior_means, posterior_log_std, nn_structure)
        derivative_mean, derivative_log_std = compute_derivative_log_posterior(posterior_means, posterior_log_std, nn_param_sample)

        log_ll = compute_log_likelihood(nn_param_sample, x_train_N, y_train_N)
        log_prior = compute_log_prior(nn_param_sample)
        log_q_posterior = compute_log_q_posterior(nn_param_sample, posterior_means, posterior_log_std)
        alpha = log_ll + log_prior - log_q_posterior

        add_two_list_dictionaries(expected_gradient_mean, derivative_mean, alpha/num_samples)
        add_two_list_dictionaries(expected_gradient_log_std, derivative_log_std, alpha/num_samples)
        #alpha = estimate_ELBO(posterior_params, x_train_N, y_train_N, 10, nn_structure)
    #print("after", expected_gradient_elbo)
    return expected_gradient_mean, expected_gradient_log_std


def compute_expected_gradient_VI_loss(posterior_params, x_train_N, y_train_N, num_samples, nn_structure):
    expected_gradient_mean, expected_gradient_log_std = compute_expected_gradient_ELBO(posterior_params,x_train_N, y_train_N, num_samples, nn_structure)
    #print("expected gradient elbo", expected_gradient_elbo)
    for layer_id, layer in enumerate(expected_gradient_mean):
        for key in layer.keys():
            layer[key] *= -1 # Flip the sign
            expected_gradient_log_std[layer_id][key] *= -1
    return expected_gradient_elbo


def do_problem_1a(outfolder):
    # Choose mtilde mean 1.0, 
    # stilde = log(0.1)
    # mhat = 0
    # shat = log(0.1)

    # Part a
    x_train_N = np.asarray([-5.0,  -2.50, 0.00, 2.50, 5.0])
    y_train_N = np.asarray([-4.91, -2.48, 0.05, 2.61, 5.09])
    num_samples = [1, 10, 100, 1000]
    #num_samples = [1]
    G = 20
    mtilde_grid_G = np.linspace(-3.0, 5.0, 20)

    nrow = 1
    ncol = len(num_samples)
    figure, axes = plt.subplots(nrow, ncol, sharex='all', sharey='all')
    nn_structure = []
    for col, ax in enumerate(axes):
        num_sample = num_samples[col]
        print("Using ", num_sample)
        vi_losses  = np.zeros((G,))
        for i,mtilde in enumerate(mtilde_grid_G):
            posterior_params = make_posterior_params_as_list_of_dicts( 
                                    n_hiddens_per_layer_list=nn_structure,\
                                    n_dims_input=1,\
                                    n_dims_output=1,\
                                    mean_weight_fill_func=lambda x : np.full(x, mtilde),\
                                    log_std_weight_full_func=lambda x: np.full(x, np.log(0.1)),
                                    mean_bias_fill_func=np.zeros,
                                    log_std_bias_fill_func=lambda x: np.full(x, np.log(0.1)))
            minus_elbo = estimate_VI_loss(posterior_params,x_train_N, y_train_N, num_sample, nn_structure)
            vi_losses[i] = minus_elbo
        ax.plot(mtilde_grid_G, vi_losses)
        ax.set_title("using " + str(num_sample) + " samples")

    outfile = os.path.join(outfolder, "p1_a.png")
    #plt.savefig(outfile)
    #plt.close()
    plt.show()


def do_problem_1c(outfolder):
    # Choose mtilde mean 1.0, 
    # stilde = log(0.1)
    # mhat = 0
    # shat = log(0.1)
    x_train_N = np.asarray([-5.0,  -2.50, 0.00, 2.50, 5.0])
    y_train_N = np.asarray([-4.91, -2.48, 0.05, 2.61, 5.09])
    # Part a
    num_samples = [1, 10, 100, 1000]
    #num_samples = [1, 10]
    G = 20
    mtilde_grid_G = np.linspace(-3.0, 5.0, 20)

    nrow = 1
    ncol = len(num_samples)
    figure, axes = plt.subplots(nrow, ncol, sharex='all', sharey='all')
    nn_structure = []
    for col, ax in enumerate(axes):
        num_sample = num_samples[col]
        print "Using ", num_sample
        vi_loss_grad_mtilde = np.zeros((G,))
        for i,mtilde in enumerate(mtilde_grid_G):
            posterior_params = make_posterior_params_as_list_of_dicts( 
                                    n_hiddens_per_layer_list=nn_structure,\
                                    n_dims_input=1,\
                                    n_dims_output=1,\
                                    mean_weight_fill_func=lambda x : np.full(x, mtilde),\
                                    log_std_weight_full_func=lambda x: np.full(x, np.log(0.1)),
                                    mean_bias_fill_func=np.zeros,
                                    log_std_bias_fill_func=lambda x: np.full(x, np.log(0.1)))
            gradient_VI = compute_expected_gradient_VI_loss(posterior_params,x_train_N, y_train_N, num_sample, nn_structure)
            #print(gradient_VI)
            mtilde_grad = gradient_VI[0]["mw"][0][0]
            #print(mtilde_grad)
            vi_loss_grad_mtilde[i] = mtilde_grad # Get the mtilde parameter, dnoted as mw in dictionary

        ax.plot(mtilde_grid_G, vi_loss_grad_mtilde)
        ax.set_title("using " + str(num_sample) + " samples")

    outfile = os.path.join(outfolder, "p1_c.png")
    #plt.savefig(outfile)
    #plt.close()
    plt.show()


def compute_gradient_norm(gradient_params):
    norm_res  = 0.0
    for layer_id, layer in enumerate(gradient_params):
        for key in layer.keys():
            norm_res += np.square(np.linalg.norm(layer[key]))
    return np.sqrt(norm_res)


def learn_neural_approximate_posterior_via_gradient_descent(nn_structure, x_train_N, y_train_N, step_size, num_steps, num_samples):
    # Init posterior
    # Measure L
    posterior_means = make_nn_params_as_list_of_dicts(nn_structure, weight_fill_func=lambda x: np.random.normal(0, 1, x),\
                                                      bias_fill_func=lambda x: np.random.normal(0,1,x))
    posterior_log_std = make_nn_params_as_list_of_dicts(nn_structure, weight_fill_func=np.zeros, bias_fill_func=np.zeros)
    # compute expected gradient
    vi_losses = []
    grad_norm = []
    epsilon = 1e-6
    for step in range(num_steps):
        # Compute current loss
        vi_loss = estimate_VI_loss(posterior_means, posterior_log_std, x_train_N, y_train_N, 1000, nn_structure)
        vi_losses.append(vi_loss)
        # Compute gradient and check for convergence
        derivative_mean, derivative_log_std \
                = compute_expected_gradient_ELBO(posterior_means, posterior_log_std, x_train_N, y_train_N, num_samples, nn_structure)

        mean_norm = compute_gradient_norm(derivative_mean)
        std_log_norm = compute_gradient_norm(derivative_log_std)
        gradient_norm = np.sqrt(mean_norm**2 + std_log_norm**2)
        grad_norm.append(gradient_norm)

        print("iter: {}, vi_loss: {}, gradient norm: {}".format(step+1, np.around(vi_loss, 3), np.around(gradient_norm,3)))
        if gradient_norm < epsilon:
            break
        # v = v - eps * dL/dv
        # Update
        add_two_list_dictionaries(posterior_means, derivative_mean, 1 * step_size(step))
        add_two_list_dictionaries(posterior_log_std, derivative_log_std, 1 * step_size(step))

    # Update posterior via gradient descent
    return posterior_means, posterior_log_std, vi_losses, grad_norm


def do_problem_2a(outfolder):
    x_train_N = np.asarray([-2.,    -1.8,   -1.,  1.,  1.8,     2.])
    y_train_N = np.asarray([-3.,  0.2224,    3.,  3.,  0.2224, -3.])
    nn_structure = [10]
    step_size = 1e-5
    num_steps = 2000
    num_samples = 1000
    step_size = lambda step : 1e-5
    #step_size = lambda step : 1e-4/(step+1)
    posterior_means, posterior_log_std, vi_losses, grads = \
            learn_neural_approximate_posterior_via_gradient_descent(nn_structure, x_train_N, y_train_N, step_size, num_steps, num_samples)


#do_problem_1a("p1")
#do_problem_1c("p1")
do_problem_2a("p2")




# NOTE: 
# TODO: check log derivative formula
# 














