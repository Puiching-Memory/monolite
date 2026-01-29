from __future__ import annotations

# Ensure registry items are imported and registered

# models
from lib.models import yolo26_monodle  # noqa: F401

# losses
from lib.losses import monolite_loss  # noqa: F401

# datasets
from lib.datasets import kitti  # noqa: F401

# optimizers
from lib.optim import registry_items as _optim_registry  # noqa: F401

# schedulers
from lib.schedulers import registry_items as _sched_registry  # noqa: F401
