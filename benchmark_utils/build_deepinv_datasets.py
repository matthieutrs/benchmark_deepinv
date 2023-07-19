# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
import requests
from io import BytesIO

import numpy as np

import torch
from torch.utils.data import TensorDataset

from torchvision import transforms

import deepinv
from deepinv.utils.demo import load_dataset, load_degradation

try:
    import mrinufft
    from mrinufft.trajectories.density import voronoi
except:
    print("Could not import mrinufft. Some functions will not be available.")

def build_set3c_dataset(deg_dir=None,
                        original_data_dir=None,
                        data_dir=None,
                        img_size=32,
                        kernel_index=1,
                        std=0.01,
                        device='cpu',
                        rebuild=False):
    r"""
    Build the dataset for the set3c benchmark.

    :param deg_dir: path to the degradation directory;
    :param original_data_dir: path to the original data directory;
    :param data_dir: path to the data directory;
    :param img_size: size of the images;
    :param kernel_index: index of the kernel to chose among the 8 motion kernels from 'Levin09.mat'.
    :param std: standard deviation of the noise.
    """

    dataset_name = "set3c"
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    # Generate a motion blur operator.
    kernel_torch = load_degradation(
        "Levin09.npy", deg_dir / "kernels", kernel_index=kernel_index
    )
    kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
        0
    )  # add batch and channel dimensions
    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)

    n_channels = 3  # 3 for color images, 1 for gray-scale images
    physics = deepinv.physics.BlurFFT(
        img_size=(n_channels, img_size, img_size),
        filter=kernel_torch,
        device=device,
        noise_model=deepinv.physics.GaussianNoise(sigma=std),
    )

    # Use parallel dataloader if using a GPU to fasten training,
    # otherwise, as all computes are on CPU, use synchronous data loading.
    num_workers = 4 if torch.cuda.is_available() else 0

    n_images_max = 3  # Maximal number of images to restore from the input dataset
    # Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
    operation = "deblur_"+str(kernel_index)+"_std_"+str(std)
    measurement_dir = data_dir / dataset_name / operation

    # if self.dataset_path is None:
    # if not hasattr(self, 'dataset_path'):
    dinv_dataset_path = measurement_dir / 'dinv_dataset0.h5'

    if not dinv_dataset_path.exists() or rebuild:
        dinv_dataset_path = deepinv.datasets.generate_dataset(
            train_dataset=None,
            test_dataset=dataset,
            physics=physics,
            device=device,
            save_dir=measurement_dir,
            test_datapoints=3,
            num_workers=num_workers,
        )

    dataset = deepinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=False)

    return dataset, physics


def build_fastMRI_dataset(deg_dir=None,
                          original_data_dir=None,
                          data_dir=None,
                          device='cpu'):
    operation = "MRI"
    dataset_name = "fastmri_knee_singlecoil"
    img_size = 128

    transform = transforms.Compose([transforms.Resize(img_size)])

    test_dataset = load_dataset(
        dataset_name, original_data_dir, transform, train=False
    )

    mask = load_degradation("mri_mask_128x128.npy", deg_dir)

    # defined physics
    physics = deepinv.physics.MRI(mask=mask, device=device)

    # Use parallel dataloader if using a GPU to fasten training,
    # otherwise, as all computes are on CPU, use synchronous data loading.
    num_workers = 4 if 'cuda' in device else 0
    n_images_max = (
        900 if torch.cuda.is_available() else 5
    )  # number of images used for training
    # (the dataset has up to 973 images, however here we use only 900)

    measurement_dir = data_dir / dataset_name / operation
    dinv_dataset_path = measurement_dir / 'dinv_dataset0.h5'

    if not dinv_dataset_path.exists():
        deepinv_datasets_path = deepinv.datasets.generate_dataset(
            train_dataset=None,
            test_dataset=test_dataset,
            physics=physics,
            device=device,
            save_dir=measurement_dir,
            train_datapoints=n_images_max,
            test_datapoints=1,
            num_workers=num_workers,
            dataset_filename=dataset_name,
        )

    test_dataset = deepinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

    return test_dataset, physics


def build_MRI_NC_T1_brainweb_dataset(deg_dir=None,
                          original_data_dir=None,
                          data_dir=None,
                          device='cpu'):
    operation = "MRI_NC"
    dataset_name = "T1_brainweb_dataset"

    url = "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fdatasets&files=brainweb_t1_ICBM_1mm_subject_0.npy"
    response = requests.get(url)
    image = np.load(BytesIO(response.content))[90, ...]
    image = torch.from_numpy(image).unsqueeze(0)  #.unsqueeze(0)
    image = (image / image.max()).to(torch.complex64)

    stacked_tensors = torch.stack([image, image])
    dataset = TensorDataset(stacked_tensors)  # create your dataset


    # Create a 2D Radial trajectory for demo
    samples_loc = mrinufft.initialize_2D_radial(Nc=100, Ns=500)
    # density = voronoi(samples_loc)

    physics = deepinv.physics.MRI_NC(
        samples_loc.reshape(-1, 2), smaps=None, shape=image.shape[-2:], density=False, n_coils=1,
        n_batchs=1, n_trans=1, backend='finufft',
    )


    # Use parallel dataloader if using a GPU to fasten training,
    # otherwise, as all computes are on CPU, use synchronous data loading.
    num_workers = 4 if 'cuda' in device else 0

    measurement_dir = data_dir / dataset_name / operation
    dinv_dataset_path = measurement_dir / 'dinv_dataset0.h5'

    if not dinv_dataset_path.exists():
        deepinv_datasets_path = deepinv.datasets.generate_dataset(
            train_dataset=None,
            test_dataset=dataset,
            physics=physics,
            device=device,
            save_dir=measurement_dir,
            train_datapoints=1,
            test_datapoints=1,
            num_workers=num_workers,
            dataset_filename=dataset_name,
        )

    test_dataset = deepinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

    return test_dataset, physics