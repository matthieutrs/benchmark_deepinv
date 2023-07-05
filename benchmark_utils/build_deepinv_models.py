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

def build_DPIR_model(physics, in_channels=3, device='cpu'):

    if 'MRI' in physics.__class__.__name__:
        lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(0.001)
        n_channels = 1
    else:
        lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(physics.noise_model.sigma)
        n_channels = 3

    params_algo = {"stepsize": stepsize,
                   "g_param": sigma_denoiser,
                   "lambda": lamb}

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    if physics.__class__.__name__ == 'MRI':
        prior = ComplexPnP(denoiser=DRUNet(in_channels=1,
                                    out_channels=1,
                                    pretrained="download",
                                    train=False,
                                    device=device))
    else:
        prior = PnP(denoiser=DRUNet(in_channels=n_channels,
                                    out_channels=n_channels,
                                    pretrained="download",
                                    train=False,
                                    device=device))


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


def build_wavelet_model(physics, in_channels=3, device='cpu'):

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    level = 3
    if physics.__class__.__name__ == 'MRI':
        prior = ComplexPnP(denoiser=deepinv.models.WaveletPrior(wv="db8", level=level).to(device))
    else:
        prior = PnP(denoiser=deepinv.models.WaveletPrior(wv="db8", level=level).to(device))

    max_iter = 200


    params_algo = {"stepsize": 1.0,
                   "g_param": 0.005,
                   "lambda": 1.0}


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



def build_waveletdict_model(physics, in_channels=3, device='cpu'):

    # Do not stop algorithm with convergence criteria
    early_stop = False

    # Select the data fidelity term
    data_fidelity = L2()

    # Specify the denoising prior
    level = 3
    if physics.__class__.__name__ == 'MRI':
        prior = ComplexPnP(denoiser=deepinv.models.WaveletDict(list_wv=["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"], level=level).to(device))
    else:
        prior = PnP(denoiser=deepinv.models.WaveletDict(list_wv=["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"], level=level).to(device))

    max_iter = 200


    params_algo = {"stepsize": 1.0,
                   "g_param": 0.005,
                   "lambda": 1.0}


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


class ComplexPnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma g}(x) = \operatorname{D}_{\sigma}(x)`.


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
        out_th = torch.view_as_real(x)
        for i in range(out_th.shape[-1]):
            out_th[..., i] = self.denoiser(torch.real(out_th[..., i]), sigma_denoiser)

        return torch.view_as_complex(out_th)
