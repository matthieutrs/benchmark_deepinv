# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
import torch
import deepinv
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import Prior, PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.parameters import get_DPIR_params

def build_DPIR_model(n_channels=3, device='cpu', sigma=0.01, prior_type='PnP'):


    lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)

    params_algo = {"stepsize": stepsize,
                   "g_param": sigma_denoiser,
                   "lambda": lamb}

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    denoiser = DRUNet(in_channels=n_channels,
                                    out_channels=n_channels,
                                    pretrained="download",
                                    train=False,
                                    device=device)

    if prior_type == 'PnP':
        prior = PnP(denoiser=denoiser)
    elif prior_type == 'ComplexPnP':
        prior = ComplexPnP(denoiser=denoiser)
    elif prior_type == 'SeparablePnP':
        prior = SeparablePnP(denoiser=denoiser)


    # instantiate the algorithm class to solve the IP problem
    model = optim_builder(
        iteration="HQS",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=False,
        params_algo=params_algo,
    )

    return model


def build_wavelet_model(stepsize=1.0, sigma=0.01, prior_type='PnP', device='cpu'):

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    level = 3
    denoiser = deepinv.models.WaveletPrior(wv="db8", level=level).to(device)

    if prior_type == 'PnP':  # For wavelets, this is equivalent to SeparablePnP
        prior = PnP(denoiser=denoiser)
        iteration = "HQS"
    elif prior_type == 'ComplexPnP':
        prior = ComplexPnP(denoiser=denoiser)
        iteration = "PGD"  # Not all algos fit for complex data yet
    elif prior_type == 'SeparablePnP':
        prior = SeparablePnP(denoiser=denoiser)
        iteration = "HQS"

    max_iter = 200

    params_algo = {"stepsize": stepsize,
                   "g_param": sigma,
                   "lambda": 1.0}


    # instantiate the algorithm class to solve the IP problem
    model = optim_builder(
        iteration=iteration,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=False,
        params_algo=params_algo,
    )

    return model



def build_waveletdict_model(stepsize=1.0, sigma=0.01, prior_type='PnP', device='cpu'):

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    level = 3
    denoiser = deepinv.models.WaveletDict(list_wv=["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"],
                                          level=level).to(device)

    if prior_type == 'PnP':  # For wavelets, this is equivalent to SeparablePnP
        prior = PnP(denoiser=denoiser)
        iteration = "HQS"
    elif prior_type == 'ComplexPnP':
        prior = ComplexPnP(denoiser=denoiser)
        iteration = "PGD"  # Not all algos fit for complex data yet
    elif prior_type == 'SeparablePnP':
        prior = SeparablePnP(denoiser=denoiser)
        iteration = "HQS"

    max_iter = 200

    params_algo = {"stepsize": stepsize,
                   "g_param": sigma,
                   "lambda": 1.0}

    # instantiate the algorithm class to solve the IP problem
    model = optim_builder(
        iteration=iteration,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=False,
        params_algo=params_algo,
    )

    return model


class ComplexPnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma g}(x) = \operatorname{D}_{\sigma}(x)`.

    Denoiser for complex data applied independently to the real and imaginary parts of the input tensor
    of shape (B, C, W, H) with complex values.


    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def prox(self, x, gamma, sigma_denoiser, *args, **kwargs):
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`g` at :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.tensor) proximity operator at :math:`x`.
        """
        out_th = torch.view_as_real(x).to(torch.float)
        for i in range(out_th.shape[-1]):
            out_th[..., i] = self.denoiser(torch.real(out_th[..., i]), sigma_denoiser)

        return torch.view_as_complex(out_th).to(torch.complex64)


class SeparablePnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma g}(x) = \operatorname{D}_{\sigma}(x)`.

    Separable proximity operator applied to each channel C of the input tensor of shape (B, C, W, H) independently.


    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def prox(self, x, gamma, sigma_denoiser, *args, **kwargs):
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`g` at :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.tensor) proximity operator at :math:`x`.
        """
        out = x.clone()
        for i in range(out.shape[1]):
            out[:, i:i+1, ...] = self.denoiser(x[:, i:i+1, ...], sigma_denoiser)

        return out
