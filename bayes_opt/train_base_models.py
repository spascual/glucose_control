import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model

from utils import noise_std


def generate_train_dataset(objective, bounds, task_list, training_size=20):
    # Sample data for each base task
    data_by_task = {}
    for task in task_list:
        # draw points from a sobol sequence
        raw_x = draw_sobol_samples(bounds=bounds,
                                   n=training_size,
                                   q=1,
                                   seed=task + 5397923).squeeze(1)
        # get observed values
        f_x = objective(raw_x, task)
        train_y = f_x + noise_std * torch.randn_like(f_x)
        train_yvar = torch.full_like(train_y, noise_std ** 2)
        # store training data
        data_by_task[task] = {
            # scale x to [0, 1]
            'train_x': normalize(raw_x, bounds=bounds),
            'train_y': train_y,
            'train_yvar': train_yvar,
        }
    return data_by_task


def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
    """
    Get a single task GP. The model will be fit unless a state_dict with model
        hyperparameters is provided.
    """
    Y_mean = train_Y.mean(dim=-2, keepdim=True)
    Y_std = train_Y.std(dim=-2, keepdim=True)
    model = FixedNoiseGP(train_X, (train_Y - Y_mean) / Y_std, train_Yvar)
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)
    return model


def fit_base_models(objective, bounds, task_list, training_size=20,
                    save_models=False, default_path='bayes_opt/base_models'):
    data_by_task = generate_train_dataset(objective, bounds, task_list, training_size)
    # Fit base model
    base_model_list = []
    for task in task_list:
        print(f"Fitting base model {task}")
        model = get_fitted_model(
            data_by_task[task]['train_x'],
            data_by_task[task]['train_y'],
            data_by_task[task]['train_yvar'],
        )
        if save_models:
            num_meals = bounds.shape[1] - 1
            path = default_path + r'/{0}_meals/{1}.pth'.format(num_meals, task)
            torch.save(model.state_dict(), path)
        base_model_list.append(model)
    return base_model_list


def load_base_models(objective, bounds, training_tasks, day):
    base_model_list = []
    data_task = generate_train_dataset(objective, bounds,
                                       task_list=[day], training_size=20)
    num_meals = bounds.shape[1] - 1
    for task in training_tasks:
        path = r'bayes_opt/base_models/{0}_meals/{1}.pth'.format(num_meals, task)
        loaded_state_dict = torch.load(path)
        model = get_fitted_model(
            data_task[day]['train_x'],
            data_task[day]['train_y'],
            data_task[day]['train_yvar'],
            state_dict=loaded_state_dict
        )
        base_model_list.append(model)
    return base_model_list
