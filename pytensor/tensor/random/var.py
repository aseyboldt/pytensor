import copy

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable, shared_constructor
from pytensor.tensor.random.type import random_generator_type, random_state_type


class RandomStateSharedVariable(SharedVariable):
    def __str__(self):
        return self.name or "RandomStateSharedVariable({})".format(repr(self.container))


class RandomGeneratorSharedVariable(SharedVariable):
    def __str__(self):
        return self.name or "RandomGeneratorSharedVariable({})".format(
            repr(self.container)
        )


@shared_constructor.register(np.random.RandomState)
@shared_constructor.register(np.random.Generator)
def randomgen_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False
):
    r"""`SharedVariable` constructor for NumPy's `Generator` and/or `RandomState`."""
    if isinstance(value, np.random.RandomState):
        rng_sv_type = RandomStateSharedVariable
        rng_type = random_state_type
    elif isinstance(value, np.random.Generator):
        rng_sv_type = RandomGeneratorSharedVariable
        rng_type = random_generator_type

    if not borrow:
        value = copy.deepcopy(value)

    return rng_sv_type(
        type=rng_type,
        value=value,
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )
