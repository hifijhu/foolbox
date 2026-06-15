from typing import Any, Optional, Union

import eagerpy as ep

from foolbox.attacks.base import FixedEpsilonAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import verify_input_bounds
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import linf
from foolbox.models.base import Model


class BayesianConfidenceAttack(FixedEpsilonAttack):
    """Standalone Foolbox-style Bayesian confidence attack.

    This is an iterative Linf attack that repeatedly samples the model,
    averages the predictive probabilities, and optimizes a combination of
    untargeted adversarial loss and low predictive entropy.
    """

    distance = linf

    def __init__(
        self,
        *,
        num_samples: int = 10,
        lambda_param: float = 0.5,
        steps: int = 20,
        rel_stepsize: float = 1.0 / 20.0,
        abs_stepsize: Optional[float] = None,
        random_start: bool = False,
    ):
        if num_samples < 1:
            raise ValueError(f"expected num_samples >= 1, got {num_samples}")
        if steps < 1:
            raise ValueError(f"expected steps >= 1, got {steps}")

        self.num_samples = num_samples
        self.lambda_param = lambda_param
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.random_start = random_start

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if not isinstance(criterion_, Misclassification):
            raise ValueError("BayesianConfidenceAttack only supports untargeted attacks")

        labels = criterion_.labels
        min_, max_ = model.bounds

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)
            x = ep.clip(x, min_, max_)
        else:
            x = x0

        rows = range(len(x0))

        def loss_fn(perturbed: ep.Tensor) -> ep.Tensor:
            avg_probs = self._mean_probabilities(model, perturbed)
            true_class_probs = avg_probs[rows, labels]
            nll = -ep.log(true_class_probs + 1e-12).sum()
            entropy = -(avg_probs * ep.log(avg_probs + 1e-12)).sum(axis=-1).sum()
            return nll - self.lambda_param * entropy

        for _ in range(self.steps):
            _, gradients = ep.value_and_grad(loss_fn, x)
            x = x + stepsize * gradients.sign()
            x = x0 + ep.clip(x - x0, -epsilon, epsilon)
            x = ep.clip(x, min_, max_)

        return restore_type(x)

    def _mean_probabilities(self, model: Model, inputs: ep.Tensor) -> ep.Tensor:
        probs = []
        for _ in range(self.num_samples):
            logits = model(inputs)
            probs.append(ep.softmax(logits, axis=-1))
        return ep.stack(probs).mean(axis=0)
