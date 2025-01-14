from typing import TYPE_CHECKING, Any

import numpy as np

import pytensor
from pytensor.link.basic import JITLinker


if TYPE_CHECKING:
    from pytensor.graph.basic import Variable


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def output_filter(self, var: "Variable", out: Any) -> Any:
        if not isinstance(var, np.ndarray) and isinstance(
            var.type, pytensor.tensor.TensorType
        ):
            return np.asarray(out, dtype=var.type.dtype)

        return out

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.numba.dispatch import numba_funcify

        return numba_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        import numba

        jitted_fn = numba.njit(fn)
        return jitted_fn

    def create_thunk_inputs(self, storage_map):
        from numpy.random import RandomState

        from pytensor.link.numba.dispatch import numba_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], RandomState):
                new_value = numba_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                # We need to remove the reference-based connection to the
                # original `RandomState`/shared variable's storage, because
                # subsequent attempts to use the same shared variable within
                # other non-Numba-fied graphs will have problems.
                sinput = [new_value]
            thunk_inputs.append(sinput)

        return thunk_inputs
