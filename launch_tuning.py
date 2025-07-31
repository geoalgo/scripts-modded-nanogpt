import logging
from pathlib import Path

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.optimizer.baselines import ASHACQR

cooldown_frac: float = 0.4  # fraction of training for learning rate cooldown
adam_lr: float = 0.008
adam_beta1: float = 0.8
adam_beta2: float = 0.95
muon_lr: float = 0.05
muon_momentum: float = 0.95

config_space = {
    "cooldown_frac": uniform(cooldown_frac - 0.3, cooldown_frac + 0.3),
    "adam_lr": loguniform(adam_lr / 10, adam_lr * 10),
    "adam_beta1": uniform(0.5, 0.99),
    "adam_beta2": uniform(0.7, 0.99),
    "muon_lr": loguniform(muon_lr / 10, muon_lr * 10),
    "muon_momentum": uniform(0.7, 0.999),
}

# adam_head_lr: float = 0.22  # learning rate for the language model head
# adam_embed_lr: float = 0.6  # learning rate for embeddings
# adam_scalar_lr: float = 0.04  # learning rate for scalar parameters

scheduler = ASHACQR(
    config_space,
    metric="val_loss",
    time_attr="st_worker_iter",
    # 1770 / 125
    max_t=15,
    points_to_evaluate=[
        {
            "cooldown_frac": cooldown_frac,
            "adam_lr": adam_lr,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "muon_lr": muon_lr,
            "muon_momentum": muon_momentum,
        }
    ],
)

logging.getLogger().setLevel(logging.DEBUG)
# TODO Handle nan
entry_point = str(Path(__file__).parent / "train_gpt.py")
tuner = Tuner(
    trial_backend=LocalBackend(
        entry_point=entry_point,
        binary="torchrun --standalone --nproc_per_node=4",
        rotate_gpus=False,
    ),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_wallclock_time=3600 * 12),
    n_workers=1,  # how many trials are evaluated in parallel
)
tuner.run()
