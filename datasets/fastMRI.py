from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path
    from torch.utils.data import DataLoader
    from benchmark_utils import build_fastMRI_dataset

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
DEG_DIR = BASE_DIR / "degradations"

# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "fastMRI"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'acceleration': [1]
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        # self.dataset_path = None

        dataset, physics = build_fastMRI_dataset(deg_dir=DEG_DIR,
                                               original_data_dir=ORIGINAL_DATA_DIR,
                                               data_dir=DATA_DIR,
                                               device='cpu')

        dataloader = DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=False
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(dataloader=dataloader, physics=physics)
