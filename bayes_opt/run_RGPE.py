import torch
import numpy as np
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf

# suppress GPyTorch warnings about adding jitter
import warnings

from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from bayes_opt.RGPE_model import compute_rank_weights, RGPE
from bayes_opt.train_base_models import get_fitted_model
from utils import noise_std

warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)

N_BATCH = 1
N_TRIALS = 1

RANDOM_INITIALIZATION_SIZE = 5

NUM_POSTERIOR_SAMPLES = 256
MC_SAMPLES = 512
N_RESTART_CANDIDATES = 512
N_RESTARTS = 10
Q_BATCH_SIZE = 1


def run_RGPE(test_task: int, objective, bounds, base_model_list):
    input_dim = bounds.shape[1]
    best_rgpe_all = []
    best_argmax_rgpe_all = []
    # Average over multiple trials
    for trial in range(N_TRIALS):
        print(f"Trial {trial + 1} of {N_TRIALS}")
        best_BMs = []
        best_rgpe = []
        # Initial random observations
        raw_x = draw_sobol_samples(bounds=bounds,
                                   n=RANDOM_INITIALIZATION_SIZE,
                                   q=1,
                                   seed=trial).squeeze(1)
        train_x = normalize(raw_x, bounds=bounds)
        train_y_noiseless = objective(raw_x, shift=test_task)
        train_y = train_y_noiseless + noise_std * torch.randn_like(train_y_noiseless)
        train_yvar = torch.full_like(train_y, noise_std ** 2)
        # keep track of the best observed point at each iteration
        best_value = train_y.max().item()
        best_rgpe.append(best_value)

        # Run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(N_BATCH):
            target_model = get_fitted_model(train_x, train_y, train_yvar)
            model_list = base_model_list + [target_model]
            rank_weights = compute_rank_weights(
                train_x,
                train_y,
                base_model_list,
                target_model,
                NUM_POSTERIOR_SAMPLES,
            )

            # create model and acquisition function
            rgpe_model = RGPE(model_list, rank_weights)
            sampler_qnei = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qNEI = qNoisyExpectedImprovement(
                model=rgpe_model,
                X_baseline=train_x,
                sampler=sampler_qnei,
            )

            # optimize
            candidate, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=bounds,
                q=Q_BATCH_SIZE,
                num_restarts=N_RESTARTS,
                raw_samples=N_RESTART_CANDIDATES,
            )

            # fetch the new values
            new_x = candidate.detach()
            new_y_noiseless = objective(unnormalize(new_x, bounds=bounds),
                                        shift=test_task)
            new_y = new_y_noiseless + noise_std * torch.randn_like(new_y_noiseless)
            new_yvar = torch.full_like(new_y, noise_std ** 2)

            # update training points
            train_x = torch.cat((train_x, new_x))
            train_y = torch.cat((train_y, new_y))
            train_yvar = torch.cat((train_yvar, new_yvar))

            # get the new best observed value
            best_value = train_y.max().item()
            best_idx = torch.argmax(train_y).item()
            best_candidate = train_x[best_idx].view(1, -1)
            _, best_BM = objective(unnormalize(best_candidate, bounds=bounds),
                                   shift=test_task,
                                   include_BMs=True)
            best_rgpe.append(best_value)
            best_BMs.append(best_BM)

        best_rgpe_all.append(best_rgpe)
        best_argmax_rgpe_all.append(best_BMs)
    BM_winner_idx = np.argmax(np.array(best_rgpe_all)[:, -1], axis=0)
    BM_winner = np.reshape(np.array(best_argmax_rgpe_all[BM_winner_idx][-1]), (2, input_dim))
    return BM_winner
