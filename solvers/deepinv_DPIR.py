from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    from benchmark_utils import build_DPIR_model as build_model
    from benchopt.stopping_criterion import SingleRunCriterion

    import torch
    from deepinv.utils import plot


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'DPIR'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
    }
    stopping_criterion = SingleRunCriterion()

    def set_objective(self, dataloader, physics):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dataloader, self.physics = dataloader, physics

    def run(self, n_iter, plot_results=False):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        # load specific parameters for DPIR
        if self.physics.__class__.__name__ == 'MRI':  # If fastMRI dataset, images are (B, 2, H, W) real
            n_channels = 1
            sigma = 0.002
            prior_type = 'SeparablePnP'
        elif self.physics.__class__.__name__ == 'MRI_NC':  # If MRI_NC dataset, images are (B, 1, H, W) complex
            n_channels = 1
            sigma = 0.01
            prior_type = 'ComplexPnP'
        else:  # else images are (B, C, H, W) real
            n_channels = 3
            sigma = self.physics.noise_model.sigma.item()
            prior_type = 'PnP'

        model = build_model(device='cpu', sigma=sigma, n_channels=n_channels, prior_type=prior_type)

        X_rec_list = []
        for batch in self.dataloader:
            X, y = batch
            X_rec = model(y, self.physics)
            X_rec_list.append(torch.real(X_rec))

        self.X_rec_list = X_rec_list

        # if plot_results:  # plot results
        #     print(y.shape, X.shape)
        #     imgs = [y, X_rec, X]
        #     name_imgs = ["Linear", "Recons.", "GT"]
        #     plot(imgs, titles=name_imgs, save_dir='images', show=True)

        if plot_results:  # plot results

            if self.physics.__class__.__name__ == 'MRI':
                imgs = [self.visMRI(self.physics.A_dagger(y)),
                        self.visMRI(X_rec),
                        self.visMRI(X)]
            else:
                imgs = [y, X_rec, X]

            name_imgs = ["Obs.", "Recons.", "GT"]
            plot(imgs, titles=name_imgs, save_dir='images', show=True)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.X_rec_list

    def visMRI(self, X):
        return torch.sqrt(X[:, 0:1, ...] ** 2 + X[:, 1:2, ...] ** 2)
