from copy import copy
from typing import List, Tuple

import numpy as np

import pytensor.tensor.basic
from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.null_type import NullType
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.basic import failure_code
from pytensor.link.c.op import COp, ExternalCOp, OpenMPOp
from pytensor.link.c.params_type import ParamsType
from pytensor.misc.frozendict import frozendict
from pytensor.misc.safe_asarray import _asarray
from pytensor.printing import FunctionPrinter, Printer, pprint
from pytensor.scalar import get_scalar_type
from pytensor.scalar.basic import bool as scalar_bool
from pytensor.scalar.basic import identity as scalar_identity
from pytensor.scalar.basic import transfer_type, upcast
from pytensor.tensor import _get_vector_length, as_tensor_variable
from pytensor.tensor import elemwise_cgen as cgen
from pytensor.tensor import get_vector_length
from pytensor.tensor.type import (
    TensorType,
    continuous_dtypes,
    discrete_dtypes,
    float_dtypes,
    lvector,
)
from pytensor.tensor.var import TensorVariable
from pytensor.utils import uniq


_numpy_ver = [int(n) for n in np.__version__.split(".")[:2]]


class DimShuffle(ExternalCOp):
    """
    Allows to reorder the dimensions of a tensor or insert or remove
    broadcastable dimensions.

    In the following examples, 'x' means that we insert a broadcastable
    dimension and a numerical index represents the dimension of the same
    rank in the tensor passed to perform.

    Parameters
    ----------
    input_broadcastable
        The expected broadcastable pattern of the input
    new_order
        A list representing the relationship between the input's
        dimensions and the output's dimensions. Each element of the
        list can either be an index or 'x'. Indices must be encoded
        as python integers, not pytensor symbolic integers.
    inplace : bool, optional
        If True (default), the output will be a view of the input.

    Notes
    -----
    If `j = new_order[i]` is an index, the output's ith dimension
    will be the input's jth dimension.
    If `new_order[i]` is `x`, the output's ith dimension will
    be 1 and broadcast operations will be allowed to do broadcasting
    over that dimension.

    If `input.type.shape[i] != 1` then `i` must be found in `new_order`.
    Broadcastable dimensions, on the other hand, can be discarded.

    .. code-block:: python

        DimShuffle((False, False, False), ['x', 2, 'x', 0, 1])

    This `Op` will only work on 3d tensors with no broadcastable
    dimensions.  The first dimension will be broadcastable,
    then we will have the third dimension of the input tensor as
    the second of the resulting tensor, etc. If the tensor has
    shape (20, 30, 40), the resulting tensor will have dimensions
    (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)

    .. code-block:: python

        DimShuffle((True, False), [1])

    This `Op` will only work on 2d tensors with the first dimension
    broadcastable.
    The second dimension of the input tensor will be the first dimension of
    the resulting tensor.
    If the tensor has shape (1, 20), the resulting tensor will have shape
    (20, ).

    Examples
    --------
    .. code-block:: python

        DimShuffle((), ['x'])  # make a 0d (scalar) into a 1d vector
        DimShuffle((False, False), [0, 1])  # identity
        DimShuffle((False, False), [1, 0])  # inverts the 1st and 2nd dimensions
        DimShuffle((False,), ['x', 0])  # make a row out of a 1d vector
                                        # (N to 1xN)
        DimShuffle((False,), [0, 'x'])  # make a column out of a 1d vector
                                        # (N to Nx1)
        DimShuffle((False, False, False), [2, 0, 1])  # AxBxC to CxAxB
        DimShuffle((False, False), [0, 'x', 1])  # AxB to Ax1xB
        DimShuffle((False, False), [1, 'x', 0])  # AxB to Bx1xA

    The reordering of the dimensions can be done with the numpy.transpose
    function.
    Adding, subtracting dimensions can be done with reshape.

    """

    _f16_ok = True
    check_input = False
    __props__ = ("input_broadcastable", "new_order", "inplace")
    c_func_file = "c_code/dimshuffle.c"
    c_func_name = "APPLY_SPECIFIC(cpu_dimshuffle)"

    @property
    def params_type(self):
        return ParamsType(
            shuffle=lvector,
            augment=lvector,
            transposition=lvector,
            inplace=scalar_bool,
        )

    def __init__(self, input_broadcastable, new_order):
        super().__init__([self.c_func_file], self.c_func_name)

        self.input_broadcastable = tuple(input_broadcastable)
        self.new_order = tuple(new_order)

        self.inplace = True

        for i, j in enumerate(new_order):
            if j != "x":
                if not isinstance(j, (int, np.integer)):
                    raise TypeError(
                        "DimShuffle indices must be Python ints; got "
                        f"{j} of type {type(j)}."
                    )
                if j >= len(input_broadcastable):
                    raise ValueError(
                        f"new_order[{i}] is {j}, but the input only has "
                        f"{len(input_broadcastable)} axes."
                    )
                if j in new_order[(i + 1) :]:
                    raise ValueError(
                        "The same input dimension may not appear "
                        f"twice in the list of output dimensions: {new_order}"
                    )

        # List of input dimensions to drop
        drop = []
        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                # We want to drop this dimension because it's not a value in
                # `new_order`
                if b == 1:
                    drop.append(i)
                else:
                    # We cannot drop non-broadcastable dimensions
                    raise ValueError(
                        "Cannot drop a non-broadcastable dimension: "
                        f"{input_broadcastable}, {new_order}"
                    )

        # This is the list of the original dimensions that we keep
        self.shuffle = [x for x in new_order if x != "x"]
        self.transposition = self.shuffle + drop
        # List of dimensions of the output that are broadcastable and were not
        # in the original input
        self.augment = sorted([i for i, x in enumerate(new_order) if x == "x"])

        if self.inplace:
            self.view_map = {0: [0]}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "func_files"):
            # Perhaps we are loading an old `Op` version of DimShuffle.
            # Let's just build the ExternalCOp.
            super().__init__([self.c_func_file], self.c_func_name)

    def make_node(self, _input):
        input = as_tensor_variable(_input)
        ib = tuple(s == 1 for s in input.type.shape)
        if ib != self.input_broadcastable:
            if len(ib) != len(self.input_broadcastable):
                raise TypeError(
                    "The number of dimensions of the "
                    f"input is incorrect for this op. Expected {self.input_broadcastable}, got {ib}."
                )
            for expected, b in zip(self.input_broadcastable, ib):
                if expected is True and b is False:
                    raise TypeError(
                        "The broadcastable pattern of the "
                        f"input is incorrect for this op. Expected {self.input_broadcastable}, got {ib}."
                    )
                # else, expected == b or expected is False and b is True
                # Both case are good.

        out_static_shape = []
        for dim_idx in self.new_order:
            if dim_idx == "x":
                out_static_shape.append(1)
            else:
                out_static_shape.append(input.type.shape[dim_idx])

        output = TensorType(dtype=input.type.dtype, shape=out_static_shape)()

        return Apply(self, [input], [output])

    def __str__(self):
        if self.inplace:
            return "InplaceDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)
        else:
            return "DimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def perform(self, node, inp, out, params):
        (res,) = inp
        (storage,) = out

        if not isinstance(res, (np.ndarray, np.memmap)):
            raise TypeError(res)

        res = res.transpose(self.transposition)

        shape = list(res.shape[: len(self.shuffle)])
        for augm in self.augment:
            shape.insert(augm, 1)
        res = res.reshape(shape)

        if not self.inplace:
            res = np.copy(res)

        storage[0] = np.asarray(res)

    def infer_shape(self, fgraph, node, shapes):
        (ishp,) = shapes
        # transpose
        rval = [ishp[i] for i in self.shuffle]

        # augment
        for augm in self.augment:
            rval.insert(augm, 1)
        return [rval]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self(*eval_points, return_list=True)

    def grad(self, inp, grads):

        (x,) = inp
        (gz,) = grads
        gz = as_tensor_variable(gz)
        grad_order = ["x"] * x.type.ndim
        for i, v in enumerate(self.new_order):
            if v != "x":
                grad_order[v] = i
        # Do not make the DimShuffle inplace as an optimization at the
        # canonicalization optimization phase will remove the inplace.
        # The inplace will be reintroduced automatically later in the graph.
        if inp[0].dtype in discrete_dtypes:
            return [inp[0].zeros_like(dtype=config.floatX)]
        else:
            return [
                DimShuffle(tuple(s == 1 for s in gz.type.shape), grad_order)(
                    Elemwise(scalar_identity)(gz)
                )
            ]


class DimShufflePrinter(Printer):
    def __p(self, new_order, pstate, r):
        if new_order != () and new_order[0] == "x":
            return f"{self.__p(new_order[1:], pstate, r)}"
        #            return "[%s]" % self.__p(new_order[1:], pstate, r)
        if list(new_order) == list(range(r.type.ndim)):
            return pstate.pprinter.process(r)
        if list(new_order) == list(reversed(range(r.type.ndim))):
            return f"{pstate.pprinter.process(r)}.T"
        return "DimShuffle{{{}}}({})".format(
            ", ".join(map(str, new_order)),
            pstate.pprinter.process(r),
        )

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print DimShuffle.")
        elif isinstance(r.owner.op, DimShuffle):
            ord = r.owner.op.new_order
            return self.__p(ord, pstate, r.owner.inputs[0])
        else:
            raise TypeError("Can only print DimShuffle.")


pprint.assign(DimShuffle, DimShufflePrinter())


class Elemwise(OpenMPOp):
    """Generalizes a scalar `Op` to tensors.

    All the inputs must have the same number of dimensions. When the
    `Op` is performed, for each dimension, each input's size for that
    dimension must be the same. As a special case, it can also be one
    but only if the input's `broadcastable` flag is ``True`` for that
    dimension. In that case, the tensor is (virtually) replicated
    along that dimension to match the size of the others.

    The dtypes of the outputs mirror those of the scalar `Op` that is
    being generalized to tensors. In particular, if the calculations
    for an output are done in-place on an input, the output type must
    be the same as the corresponding input type (see the doc of
    `ScalarOp` to get help about controlling the output type)

    Notes
    -----
    -``Elemwise(add)``: represents ``+`` on tensors ``x + y``
    -``Elemwise(add, {0 : 0})``: represents the ``+=`` operation ``x += y``
    -``Elemwise(add, {0 : 1})``: represents ``+=`` on the second argument ``y += x``
    -``Elemwise(mul)(np.random.random((10, 5)), np.random.random((1, 5)))``:
    the second input is completed along the first dimension to match the first input
    -``Elemwise(true_div)(np.random.random(10, 5), np.random.random(10, 1))``: same but along the
    second dimension
    -``Elemwise(int_div)(np.random.random((1, 5)), np.random.random((10, 1)))``:
    the output has size ``(10, 5)``.
    -``Elemwise(log)(np.random.random((3, 4, 5)))``

    """

    __props__ = ("scalar_op", "inplace_pattern")

    def __init__(
        self, scalar_op, inplace_pattern=None, name=None, nfunc_spec=None, openmp=None
    ):
        """

        Parameters
        ----------
        scalar_op
            An instance of a subclass of `ScalarOp` which works uniquely
            on scalars.
        inplace_pattern
            A dictionary that maps the index of an output to the
            index of an input so the output is calculated inplace using
            the input's storage. (Just like `Op.destroy_map`, but without the lists.)
        nfunc_spec
            Either ``None`` or a tuple of three elements, ``(nfunc_name, nin,
            nout)`` such that ``getattr(numpy, nfunc_name)`` implements this
            operation, takes ``nin``-many inputs and ``nout``-many outputs.  Note
            that ``nin`` cannot always be inferred from the scalar `Op`'s own
            ``nin`` field, because that value is sometimes zero (meaning a variable
            number of inputs), whereas the NumPy function may not have var-args.

        """
        assert not isinstance(scalar_op, type(self))
        if inplace_pattern is None:
            inplace_pattern = frozendict({})
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = {o: [i] for o, i in self.inplace_pattern.items()}

        if nfunc_spec is None:
            nfunc_spec = getattr(scalar_op, "nfunc_spec", None)
        self.nfunc_spec = nfunc_spec
        self.__setstate__(self.__dict__)
        super().__init__(openmp=openmp)

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop("ufunc")
        d.pop("nfunc")
        d.pop("__epydoc_asRoutine", None)
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.ufunc = None
        self.nfunc = None
        self.inplace_pattern = frozendict(self.inplace_pattern)

    def get_output_info(self, dim_shuffle, *inputs):
        """Return the outputs dtype and broadcastable pattern and the
        dimshuffled inputs.

        """
        shadow = self.scalar_op.make_node(
            *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
        )

        target_length = max(input.type.ndim for input in inputs)

        args = []
        for input in inputs:
            length = input.type.ndim
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                # TODO: use LComplete instead
                args.append(
                    dim_shuffle(
                        tuple(1 if s == 1 else None for s in input.type.shape),
                        ["x"] * difference + list(range(length)),
                    )(input)
                )
        inputs = args

        # HERE: all the broadcast dims have the same length now

        # cleverness: we iterate over the first, second, third broadcast flag
        # of all inputs in parallel... the all() gives us each output
        # broadcastable bit in turn.

        def get_most_specialized_shape(shapes):
            shapes = set(shapes)
            # All shapes are the same
            if len(shapes) == 1:
                return tuple(shapes)[0]

            # Only valid indeterminate case
            if shapes == {None, 1}:
                return None

            shapes.discard(1)
            shapes.discard(None)
            if len(shapes) > 1:
                raise ValueError
            return tuple(shapes)[0]

        # it is multiplied by nout because Elemwise supports multiple outputs
        # (nout of them)
        try:
            out_shapes = [
                [
                    get_most_specialized_shape(shape)
                    for shape in zip(*[inp.type.shape for inp in inputs])
                ]
            ] * shadow.nout
        except ValueError:
            raise ValueError(
                f"Incompatible Elemwise input shapes {[inp.type.shape for inp in inputs]}"
            )

        # inplace_pattern maps output idx -> input idx
        inplace_pattern = self.inplace_pattern
        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for out_s, in_s in zip(
                    out_shapes[overwriter],
                    inputs[overwritten].type.shape,
                ):
                    if in_s == 1 and out_s != 1:
                        raise ValueError(
                            "Operation cannot be done inplace on an input "
                            "with broadcasted dimensions."
                        )

        out_dtypes = [o.type.dtype for o in shadow.outputs]
        if any(
            inputs[i].type.dtype != out_dtypes[o] for o, i in inplace_pattern.items()
        ):
            raise TypeError(
                (
                    "Cannot do an inplace operation on incompatible data types.",
                    ([i.type.dtype for i in inputs], out_dtypes, inplace_pattern),
                )
            )
        assert len(out_dtypes) == len(out_shapes)
        return out_dtypes, out_shapes, inputs

    def make_node(self, *inputs):
        """
        If the inputs have different number of dimensions, their shape
        is left-completed to the greatest number of dimensions with 1s
        using DimShuffle.
        """
        inputs = [as_tensor_variable(i) for i in inputs]
        out_dtypes, out_shapes, inputs = self.get_output_info(DimShuffle, *inputs)
        outputs = [
            TensorType(dtype=dtype, shape=shape)()
            for dtype, shape in zip(out_dtypes, out_shapes)
        ]
        return Apply(self, inputs, outputs)

    def __str__(self):
        if self.name is None:
            if self.inplace_pattern:
                items = list(self.inplace_pattern.items())
                items.sort()
                return f"{type(self).__name__}{{{self.scalar_op}}}{items}"
            else:
                return f"{type(self).__name__}{{{self.scalar_op}}}"
        else:
            return self.name

    def R_op(self, inputs, eval_points):
        outs = self(*inputs, return_list=True)
        rval = [None for x in outs]
        # For each output
        for idx, out in enumerate(outs):
            # make such that _bgrads computes only the gradients of the
            # current output on the inputs ( and not all outputs)
            ograds = [x.zeros_like() for x in outs]
            ograds[idx] = pytensor.tensor.basic.ones_like(out)

            bgrads = self._bgrad(inputs, outs, ograds)
            rop_out = None

            for jdx, (inp, eval_point) in enumerate(zip(inputs, eval_points)):
                # if None, then we can just ignore this branch ..
                # what we do is to assume that for any non-differentiable
                # branch, the gradient is actually 0, which I think is not
                # the right thing to do .. have to talk to Ian and James
                # about it

                if bgrads[jdx] is None or isinstance(
                    bgrads[jdx].type, DisconnectedType
                ):
                    pass
                elif eval_point is not None:
                    if rop_out is None:
                        rop_out = bgrads[jdx] * eval_point
                    else:
                        rop_out = rop_out + bgrads[jdx] * eval_point

            rval[idx] = rop_out

        return rval

    def connection_pattern(self, node):

        if hasattr(self.scalar_op, "connection_pattern"):
            return self.scalar_op.connection_pattern(node)

        return [[True for output in node.outputs] for ipt in node.inputs]

    def L_op(self, inputs, outs, ograds):
        from pytensor.tensor.math import sum as at_sum

        # Compute grad with respect to broadcasted input
        rval = self._bgrad(inputs, outs, ograds)

        # TODO: make sure that zeros are clearly identifiable
        # to the gradient.grad method when the outputs have
        # some integer and some floating point outputs
        if any(out.type.dtype not in continuous_dtypes for out in outs):
            # For integer output, return value may only be zero or undefined
            # We don't bother with trying to check that the scalar ops
            # correctly returned something that evaluates to 0, we just make
            # the return value obviously zero so that gradient.grad can tell
            # this op did the right thing.
            new_rval = []
            for elem, ipt in zip(rval, inputs):
                if isinstance(elem.type, (NullType, DisconnectedType)):
                    new_rval.append(elem)
                else:
                    elem = ipt.zeros_like()
                    if str(elem.type.dtype) not in continuous_dtypes:
                        elem = elem.astype(config.floatX)
                    assert str(elem.type.dtype) not in discrete_dtypes
                    new_rval.append(elem)
            return new_rval

        # sum out the broadcasted dimensions
        for i, ipt in enumerate(inputs):
            if isinstance(rval[i].type, (NullType, DisconnectedType)):
                continue

            # List of all the dimensions that are broadcastable for input[i] so
            # we can sum over them
            # TODO: only count dimensions that were effectively broadcasted
            to_sum = [
                j
                for j, in_s in enumerate(ipt.type.shape)
                if in_s == 1 and outs[0].type.shape[j] != 1
            ]

            if to_sum:
                sr = at_sum(rval[i], axis=to_sum, keepdims=True)
                rval[i] = sr

        return rval

    def _bgrad(self, inputs, outputs, ograds):
        # returns grad, with respect to broadcasted versions of inputs

        with config.change_flags(compute_test_value="off"):

            def as_scalar(t):
                if isinstance(t.type, (NullType, DisconnectedType)):
                    return t
                return get_scalar_type(t.type.dtype)()

            scalar_inputs = list(map(as_scalar, inputs))
            scalar_ograds = list(map(as_scalar, ograds))
            scalar_outputs = self.scalar_op.make_node(
                *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
            ).outputs
            scalar_igrads = self.scalar_op.L_op(
                scalar_inputs, scalar_outputs, scalar_ograds
            )
            for igrad in scalar_igrads:
                assert igrad is not None, self.scalar_op

        if not isinstance(scalar_igrads, (list, tuple)):
            raise TypeError(
                f"{str(self.scalar_op)}.grad returned {str(type(scalar_igrads))} instead of list or tuple"
            )

        nd = inputs[0].type.ndim  # this is the same for everyone

        def transform(r):
            # From a graph of ScalarOps, make a graph of Broadcast ops.
            if isinstance(r.type, (NullType, DisconnectedType)):
                return r
            if r in scalar_inputs:
                return inputs[scalar_inputs.index(r)]
            if r in scalar_outputs:
                return outputs[scalar_outputs.index(r)]
            if r in scalar_ograds:
                return ograds[scalar_ograds.index(r)]
            node = r.owner
            if node is None:
                # the gradient contains a constant, translate it as
                # an equivalent TensorType of size 1 and proper number of
                # dimensions
                res = pytensor.tensor.basic.constant(
                    np.asarray(r.data), dtype=r.type.dtype
                )
                return DimShuffle((), ["x"] * nd)(res)

            new_r = Elemwise(node.op, {})(*[transform(ipt) for ipt in node.inputs])
            return new_r

        ret = []
        for scalar_igrad, ipt in zip(scalar_igrads, inputs):
            if scalar_igrad is None:
                # undefined gradient
                ret.append(None)
                continue
            ret.append(transform(scalar_igrad))

        return ret

    def prepare_node(self, node, storage_map, compute_map, impl):
        # Postpone the ufunc building to the last minutes due to:
        # - NumPy ufunc support only up to 31 inputs.
        #   But our c code support more.
        # - nfunc is reused for scipy and scipy is optional
        if len(node.inputs) > 32 and self.ufunc and impl == "py":
            impl = "c"

        if getattr(self, "nfunc_spec", None) and impl != "c":
            self.nfunc = getattr(np, self.nfunc_spec[0], None)
            if self.nfunc is None:
                # Not inside NumPy. So probably another package like scipy.
                symb = self.nfunc_spec[0].split(".")
                for idx in range(1, len(self.nfunc_spec[0])):
                    try:
                        module = __import__(".".join(symb[:idx]))
                    except ImportError:
                        break
                for sub in symb[1:]:
                    try:
                        module = getattr(module, sub)
                    except AttributeError:
                        module = None
                        break
                self.nfunc = module

        if (
            len(node.inputs) < 32
            and (self.nfunc is None or self.scalar_op.nin != len(node.inputs))
            and self.ufunc is None
            and impl == "py"
        ):

            ufunc = np.frompyfunc(
                self.scalar_op.impl, len(node.inputs), self.scalar_op.nout
            )
            if self.scalar_op.nin > 0:
                # We can reuse it for many nodes
                self.ufunc = ufunc
            else:
                node.tag.ufunc = ufunc

        # Numpy ufuncs will sometimes perform operations in
        # float16, in particular when the input is int8.
        # This is not something that we want, and we do not
        # do it in the C code, so we specify that the computation
        # should be carried out in the returned dtype.
        # This is done via the "sig" kwarg of the ufunc, its value
        # should be something like "ff->f", where the characters
        # represent the dtype of the inputs and outputs.

        # NumPy 1.10.1 raise an error when giving the signature
        # when the input is complex. So add it only when inputs is int.
        out_dtype = node.outputs[0].dtype
        if (
            out_dtype in float_dtypes
            and isinstance(self.nfunc, np.ufunc)
            and node.inputs[0].dtype in discrete_dtypes
        ):
            char = np.sctype2char(out_dtype)
            sig = char * node.nin + "->" + char * node.nout
            node.tag.sig = sig
        node.tag.fake_node = Apply(
            self.scalar_op,
            [
                get_scalar_type(dtype=input.type.dtype).make_variable()
                for input in node.inputs
            ],
            [
                get_scalar_type(dtype=output.type.dtype).make_variable()
                for output in node.outputs
            ],
        )

        self.scalar_op.prepare_node(node.tag.fake_node, None, None, impl)

    def perform(self, node, inputs, output_storage):
        if len(node.inputs) >= 32:
            # Some versions of NumPy will segfault, other will raise a
            # ValueError, if the number of inputs to a ufunc is 32 or more.
            # In that case, the C version should be used, or Elemwise fusion
            # should be disabled.
            super().perform(node, inputs, output_storage)

        for d, dim_shapes in enumerate(zip(*(i.shape for i in inputs))):
            if len(set(dim_shapes) - {1}) > 1:
                raise ValueError(f"Shapes on dimension {d} do not match: {dim_shapes}")

        # Determine the shape of outputs
        out_shape = []
        for values in zip(*[input.shape for input in inputs]):
            if any(v == 0 for v in values):
                # All non-broadcasted dimensions should be zero
                assert max(values) <= 1
                out_shape.append(0)
            else:
                out_shape.append(max(values))
        out_shape = tuple(out_shape)

        ufunc_args = inputs
        ufunc_kwargs = {}
        # We supported in the past calling manually op.perform.
        # To keep that support we need to sometimes call self.prepare_node
        if self.nfunc is None and self.ufunc is None:
            self.prepare_node(node, None, None, "py")
        if self.nfunc and len(inputs) == self.nfunc_spec[1]:
            ufunc = self.nfunc
            nout = self.nfunc_spec[2]
            if hasattr(node.tag, "sig"):
                ufunc_kwargs["sig"] = node.tag.sig
            # Unfortunately, the else case does not allow us to
            # directly feed the destination arguments to the nfunc
            # since it sometimes requires resizing. Doing this
            # optimization is probably not worth the effort, since we
            # should normally run the C version of the Op.
        else:
            # the second calling form is used because in certain versions of
            # numpy the first (faster) version leads to segfaults
            if self.ufunc:
                ufunc = self.ufunc
            elif not hasattr(node.tag, "ufunc"):
                # It happen that make_thunk isn't called, like in
                # get_scalar_constant_value
                self.prepare_node(node, None, None, "py")
                # prepare_node will add ufunc to self or the tag
                # depending if we can reuse it or not. So we need to
                # test both again.
                if self.ufunc:
                    ufunc = self.ufunc
                else:
                    ufunc = node.tag.ufunc
            else:
                ufunc = node.tag.ufunc

            nout = ufunc.nout

        variables = ufunc(*ufunc_args, **ufunc_kwargs)

        if nout == 1:
            variables = [variables]

        for i, (variable, storage, nout) in enumerate(
            zip(variables, output_storage, node.outputs)
        ):
            if getattr(variable, "dtype", "") == "object":
                # Since numpy 1.6, function created with numpy.frompyfunc
                # always return an ndarray with dtype object
                variable = np.asarray(variable, dtype=nout.dtype)

            if i in self.inplace_pattern:
                odat = inputs[self.inplace_pattern[i]]
                odat[...] = variable
                storage[0] = odat

            # Sometimes NumPy return a Python type.
            # Some PyTensor op return a different dtype like floor, ceil,
            # trunc, eq, ...
            elif not isinstance(variable, np.ndarray) or variable.dtype != nout.dtype:
                variable = np.asarray(variable, nout.dtype)
                # The next line is needed for numpy 1.9. Otherwise
                # there are tests that fail in DebugMode.
                # Normally we would call pytensor.misc._asarray, but it
                # is faster to inline the code. We know that the dtype
                # are the same string, just different typenum.
                if np.dtype(nout.dtype).num != variable.dtype.num:
                    variable = variable.view(dtype=nout.dtype)
                storage[0] = variable
            # numpy.real return a view!
            elif not variable.flags.owndata:
                storage[0] = variable.copy()
            else:
                storage[0] = variable

    def infer_shape(self, fgraph, node, i_shapes) -> List[Tuple[TensorVariable, ...]]:

        if len(node.outputs) > 1:
            from pytensor.tensor.exceptions import ShapeError

            raise ShapeError(
                "Multiple outputs are not supported by the default `Elemwise.infer_shape`"
            )

        out_shape = pytensor.tensor.broadcast_shape(*i_shapes, arrays_are_shapes=True)

        # The `as_tensor_variable` should convert `ScalarType`s to `TensorType`s
        return [tuple(as_tensor_variable(s) for s in out_shape)]

    def _c_all(self, node, nodename, inames, onames, sub):
        # Some `Op`s directly call `Elemwise._c_all` or `Elemwise.c_code`
        # To not request all of them to call prepare_node(), do it here.
        # There is no harm if it get called multiple times.
        if not hasattr(node.tag, "fake_node"):
            self.prepare_node(node, None, None, "c")
        _inames = inames
        _onames = onames

        inames = uniq(inames)
        inputs = uniq(node.inputs)
        # assert that inames and inputs order stay consistent.
        # This is to protect again futur change of uniq.
        assert len(inames) == len(inputs)
        ii, iii = list(zip(*uniq(list(zip(_inames, node.inputs)))))
        assert all(x == y for x, y in zip(ii, inames))
        assert all(x == y for x, y in zip(iii, inputs))

        defines = ""
        undefs = ""

        # The destroy map is a map of output indices to input indices
        # that overwrite them.  We just convert them to the actual
        # Variables.
        dmap = {
            node.outputs[o]: [node.inputs[i]] for o, i in self.inplace_pattern.items()
        }

        # dtypes of the inputs
        idtypes = [input.type.dtype_specs()[1] for input in inputs]

        # These are the outputs that we will need to allocate
        # (output, name, name of the c type), transposed
        real = list(
            zip(
                *[
                    (r, s, r.type.dtype_specs()[1])
                    for r, s in zip(node.outputs, onames)
                    if r not in dmap
                ]
            )
        )
        if real:
            real_outputs, real_onames, real_odtypes = real
        else:
            real_outputs, real_onames, real_odtypes = [], [], []

        # Outputs that are aliased with an input (inplace)
        # (output, name), transposed (c type name not needed since we don't
        # need to allocate.
        aliased = list(
            zip(*[(r, s) for (r, s) in zip(node.outputs, onames) if r in dmap])
        )
        if aliased:
            aliased_outputs, aliased_onames = aliased
        else:
            aliased_outputs, aliased_onames = [], []

        # for each input:
        # same as range(ndim), but with 'x' at all broadcastable positions
        orders = [
            [s == 1 and "x" or i for i, s in enumerate(input.type.shape)]
            for input in inputs
        ]

        # number of nested loops we will need (all inputs have same
        # dimensionality)
        nnested = len(orders[0])
        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(inputs, inames)):
            # the c generators will substitute the input names for
            # references to loop variables lv0, lv1, ...
            sub[f"lv{i}"] = iname

        decl = cgen.make_declare(orders, idtypes, sub)
        checks = cgen.make_checks(orders, idtypes, sub)

        # Check if all inputs (except broadcasted scalar) are fortran.
        # In that case, create a fortran output ndarray.
        z = list(zip(inames, inputs))
        alloc_fortran = " && ".join(
            [
                f"PyArray_ISFORTRAN({arr})"
                for arr, var in z
                if not all(s == 1 for s in var.type.shape)
            ]
        )
        # If it is a scalar, make it c contig to prevent problem with
        # NumPy C and F contig not always set as both of them.
        if len(alloc_fortran) == 0:
            alloc_fortran = "0"

        alloc = ""
        # We loop over the "real" outputs, i.e., those that are not
        # inplace (must be allocated) and we declare/allocate/check
        # them
        for output, oname, odtype in zip(real_outputs, real_onames, real_odtypes):
            i += 1  # before this loop, i = number of inputs
            sub[f"lv{i}"] = oname
            sub["olv"] = oname
            alloc += cgen.make_declare(
                [list(range(nnested))], [odtype], dict(sub, lv0=oname)
            )
            alloc += cgen.make_alloc(orders, odtype, sub, fortran=alloc_fortran)
            alloc += cgen.make_checks(
                [list(range(nnested))], [odtype], dict(sub, lv0=oname)
            )
        olv_index = i  # index of the last output

        # We loop over the "aliased" outputs, i.e., those that are
        # inplace (overwrite the contents of one of the inputs) and
        # make the output pointers point to their corresponding input
        # pointers.
        for output, oname in zip(aliased_outputs, aliased_onames):
            olv_index = inputs.index(dmap[output][0])
            iname = inames[olv_index]
            # We make the output point to the corresponding input and
            # decrease the reference of whatever the output contained
            # prior to this
            alloc += (
                """
            if (%(oname)s) {
                Py_XDECREF(%(oname)s);
            }
            %(oname)s = %(iname)s;
            Py_XINCREF(%(oname)s);
            """
                % locals()
            )
            # We alias the scalar variables
            defines += f"#define {oname}_i {iname}_i\n"
            undefs += f"#undef {oname}_i\n"

        # Note: here, olv_index is either the index of the last output
        # which is allocated, OR, if there are any aliased outputs,
        # the index of the last of these aliased outputs.

        # We generate the C code of the inner loop using the scalar op
        if self.openmp:
            # If we are using openmp, we need to get rid of the "goto"
            # statement in sub['fail']. For now we recreate it here.
            fail = failure_code(sub, use_goto=False)
        else:
            fail = sub["fail"]
        task_code = self.scalar_op.c_code(
            node.tag.fake_node,
            nodename + "_scalar_",
            [f"{s}_i" for s in _inames],
            [f"{s}_i" for s in onames],
            dict(sub, fail=fail),
        )
        code = (
            """
        {
            %(defines)s
            %(task_code)s
            %(undefs)s
        }
        """
            % locals()
        )

        loop_orders = orders + [list(range(nnested))] * len(real_onames)
        dtypes = idtypes + list(real_odtypes)
        if all(
            [o.ndim <= 1 for o in node.outputs]
            or
            # Use simpler code when output ndim == 0 or 1
            # or for broadcated scalar.
            all(s == 1 for s in node.outputs[0].type.shape)
        ):
            if nnested:
                all_code = [("", "")] * (nnested - 1) + [("", code)] + [""]
            else:
                all_code = [code]
            if len(all_code) == 1:
                # No loops
                task_decl = "".join(
                    [
                        "{}& {}_i = *{}_iter;\n".format(dtype, name, name)
                        for name, dtype in zip(
                            inames + list(real_onames), idtypes + list(real_odtypes)
                        )
                    ]
                )

                preloops = {}
                for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
                    for j, index in enumerate(loop_order):
                        if index != "x":
                            preloops.setdefault(j, "")
                            preloops[j] += (
                                "%%(lv%(i)s)s_iter = (%(dtype)s*)"
                                "(PyArray_DATA(%%(lv%(i)s)s));\n" % locals()
                            ) % sub
                            break
                    else:  # all broadcastable
                        preloops.setdefault(0, "")
                        preloops[0] += (
                            "%%(lv%(i)s)s_iter = (%(dtype)s*)"
                            "(PyArray_DATA(%%(lv%(i)s)s));\n" % locals()
                        ) % sub

                init_array = preloops.get(0, " ")
                loop = (
                    """
                {
                  %(defines)s
                  %(init_array)s
                  %(task_decl)s
                  %(task_code)s
                  %(undefs)s
                }
                """
                    % locals()
                )
            else:
                loop = cgen.make_loop(
                    loop_orders=loop_orders,
                    dtypes=dtypes,
                    loop_tasks=all_code,
                    sub=sub,
                    openmp=self.openmp,
                )
        else:
            loop = cgen.make_reordered_loop(
                init_loop_orders=loop_orders,
                olv_index=olv_index,
                dtypes=dtypes,
                inner_task=code,
                sub=sub,
                openmp=self.openmp,
            )

        # If all inputs and outputs are contiguous
        # and the scalar op define optimized code for that case
        # use it! The scalar_op needs to check the type-level shapes itself.
        if (
            all(o.ndim >= 1 for o in node.outputs)
            and
            # Don't use the contig code for broadcasted scalar.
            not all(s == 1 for s in node.outputs[0].type.shape)
        ):
            contig = None
            try:
                contig = self.scalar_op.c_code_contiguous(
                    node, nodename + "_scalar_contig_", _inames, onames, sub
                )
            except MethodNotDefined:
                # Try to make one generic version, this will help the
                # compiler to vectorize the code as their won't be as
                # many ptr and the stride will be hard coded.
                if all(
                    # io.type.shape == node.outputs[1].type.shape
                    # Elemwise does not specify non-broadcastable static/type-levelshape
                    # information for its outputs yet
                    node.outputs[0].type.is_super(io.type)
                    for io in node.inputs + node.outputs
                ) and (
                    len(node.inputs) <= 1
                    # If either one of the inputs has a `None` shape, we cannot
                    # assume they will have the same size
                    or all(
                        len(set(inp_shape)) == 1 and None not in inp_shape
                        for inp_shape in zip(*(inp.type.shape for inp in node.inputs))
                    )
                ):
                    z = onames[0]
                    contig = f"""
                    // All output have the same size
                    npy_intp n = PyArray_SIZE({z});
                    """
                    index = ""
                    for x, var in zip(inames + onames, inputs + node.outputs):
                        if not all(s == 1 for s in var.type.shape):
                            contig += (
                                """
            dtype_%(x)s * %(x)s_ptr = (dtype_%(x)s*) PyArray_DATA(%(x)s);
                            """
                                % locals()
                            )
                            index += (
                                """
            dtype_%(x)s& %(x)s_i = %(x)s_ptr[i];
                            """
                                % locals()
                            )
                        else:
                            contig += (
                                """
            dtype_%(x)s& %(x)s_i = ((dtype_%(x)s*) PyArray_DATA(%(x)s))[0];
                            """
                                % locals()
                            )
                    if self.openmp:
                        contig += f"""#pragma omp parallel for if(n>={int(config.openmp_elemwise_minsize)})
                        """
                    contig += (
                        """
                    for(int i=0; i<n; i++){
                        %(index)s
                        %(task_code)s;
                    }
                    """
                        % locals()
                    )
            if contig is not None:
                z = list(zip(inames + onames, inputs + node.outputs))
                all_broadcastable = all(s == 1 for s in var.type.shape)
                cond1 = " && ".join(
                    [
                        "PyArray_ISCONTIGUOUS(%s)" % arr
                        for arr, var in z
                        if not all_broadcastable
                    ]
                )
                cond2 = " && ".join(
                    [
                        "PyArray_ISFORTRAN(%s)" % arr
                        for arr, var in z
                        if not all_broadcastable
                    ]
                )
                loop = (
                    """
            if((%(cond1)s) || (%(cond2)s)){
                %(contig)s
            }else{
                %(loop)s
            }
            """
                    % locals()
                )
        return decl, checks, alloc, loop, ""

    def c_code(self, node, nodename, inames, onames, sub):
        if (
            any(i.dtype == "float16" for i in node.inputs)
            or any(o.dtype == "float16" for o in node.outputs)
            or
            # This is for Composite
            getattr(self.scalar_op, "inner_float16", False)
        ):
            # Disable C code for float16 vars
            raise NotImplementedError()
        code = "\n".join(self._c_all(node, nodename, inames, onames, sub))
        return code

    def c_headers(self, **kwargs):
        return ["<vector>", "<algorithm>"]

    def c_header_dirs(self, **kwargs):
        return self.scalar_op.c_header_dirs(**kwargs)

    def c_support_code(self, **kwargs):
        return self.scalar_op.c_support_code(**kwargs)

    def c_support_code_apply(self, node, nodename):
        support_code = self.scalar_op.c_support_code_apply(node, nodename + "_scalar_")
        return support_code

    def c_code_cache_version_apply(self, node):
        version = [14]  # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(
            self.scalar_op,
            [
                get_scalar_type(dtype=input.type.dtype).make_variable()
                for input in node.inputs
            ],
            [
                get_scalar_type(dtype=output.type.dtype).make_variable()
                for output in node.outputs
            ],
        )
        version.append(self.scalar_op.c_code_cache_version_apply(scalar_node))
        for i in node.inputs + node.outputs:
            version.append(get_scalar_type(dtype=i.type.dtype).c_code_cache_version())
        version.append(("openmp", self.openmp))
        if all(version):
            return tuple(version)
        else:
            return ()


class CAReduce(COp):
    """Reduces a scalar operation along specified axes.

    The scalar op should be both commutative and associative.

    `CAReduce` = Commutative Associative Reduce.

    The output will have the same shape as the input minus the reduced
    dimensions. It will contain the variable of accumulating all values
    over the reduced dimensions using the specified scalar `Op`.

    Notes
    -----
    .. code-block:: python

        CAReduce(add)      # sum (ie, acts like the numpy sum operation)
        CAReduce(mul)      # product
        CAReduce(maximum)  # max
        CAReduce(minimum)  # min
        CAReduce(or_)      # any # not lazy
        CAReduce(and_)     # all # not lazy
        CAReduce(xor)      # a bit at 1 tell that there was an odd number of
                           # bit at that position that where 1. 0 it was an
                           # even number ...

    In order to (eventually) optimize memory usage patterns,
    `CAReduce` makes zero guarantees on the order in which it
    iterates over the dimensions and the elements of the
    array(s). Therefore, to ensure consistent variables, the scalar
    operation represented by the reduction must be both commutative
    and associative (eg add, multiply, maximum, binary or/and/xor - but not
    subtract, divide or power).

    """

    __props__ = ("scalar_op", "axis", "dtype", "acc_dtype", "upcast_discrete_output")

    def __init__(
        self,
        scalar_op,
        axis=None,
        dtype=None,
        acc_dtype=None,
        upcast_discrete_output=False,
    ):
        """

        Parameters
        ----------
        scalar_op
            A binary scalar `Op` with only one output.
            It must be commutative and associative.
        axis
            - the dimension along which we want to reduce
            - list of dimensions that we want to reduce
            - if ``None``, all dimensions are reduced
        dtype
            The dtype of the returned tensor. If ``None``, then we use the default
            dtype which is the same as the input array's dtype except when
            `upcast_discrete_output` is ``True`` and the following holds:

            - the input dtype is a signed integer of precision < 64 bit, in which
            case we use int64
            - the input dtype is an unsigned integer of precision < 64 bit, in
            which case we use uint64

            This default dtype does _not_ depend on the value of `acc_dtype`.
            This behavior is similar in spirit to that of NumPy, except that
            NumPy uses the default machine integer while we always use 64 bit
            integers to avoid platform-dependent behavior.
        acc_dtype
            The dtype of the internal accumulator.
            If ``None`` (default), we use the dtype in the list below,
            or the input dtype if its precision is higher:

            - for int dtypes, we use at least int64;
            - for uint dtypes, we use at least uint64;
            - for float dtypes, we use at least float64;
            - for complex dtypes, we use at least complex128.
        upcast_discrete_output
            See

        """
        if scalar_op.nin not in (-1, 2) or scalar_op.nout != 1:
            raise NotImplementedError(
                "CAReduce only supports binary functions with a single output."
            )

        self.axis = None
        self.scalar_op = scalar_op

        if axis is not None:
            if isinstance(axis, (int, np.integer)) or (
                isinstance(axis, np.ndarray) and not axis.shape
            ):
                self.axis = (int(axis),)
            else:
                self.axis = tuple(axis)

        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.upcast_discrete_output = upcast_discrete_output

    @property
    def ufunc(self):
        if hasattr(self, "_ufunc"):
            return self._ufunc

        if hasattr(self.scalar_op, "nfunc_spec") and hasattr(
            np, self.scalar_op.nfunc_spec[0]
        ):
            self._ufunc = getattr(np, self.scalar_op.nfunc_spec[0])
        else:
            self._ufunc = np.frompyfunc(
                self.scalar_op.impl, 2, 1, identity=self.scalar_op.identity
            )

        return self._ufunc

    def _output_dtype(self, idtype):

        if not self.upcast_discrete_output:
            return idtype

        dtype = self.dtype

        if dtype == "OLD":
            return dict(
                int8="int32",
                int16="int32",
                int32="int64",
                uint8="uint32",
                uint16="uint32",
                uint32="uint64",
            ).get(idtype, idtype)
        elif dtype is None:
            # If input has a discrete dtype, upcast it to 64
            return dict(
                bool="int64",
                int8="int64",
                int16="int64",
                int32="int64",
                uint8="uint64",
                uint16="uint64",
                uint32="uint64",
            ).get(idtype, idtype)
        else:
            # The important is that the accumulator dtype does not
            # lose precision. Then, the result can be downcasted.
            return dtype

    def _acc_dtype(self, idtype):
        acc_dtype = self.acc_dtype
        if acc_dtype is None:
            return dict(
                bool="int64",
                int8="int64",
                int16="int64",
                int32="int64",
                uint8="uint64",
                uint16="uint64",
                uint32="uint64",
                float16="float32",
                float32="float64",
                complex64="complex128",
            ).get(idtype, idtype)
        elif acc_dtype in continuous_dtypes and idtype in discrete_dtypes:
            # Specifying a continuous accumulator for discrete input is OK
            return acc_dtype
        else:
            # The conversion has to be considered an upcast.
            upcasted_dtype = upcast(idtype, acc_dtype)
            if acc_dtype != upcasted_dtype:
                raise TypeError(
                    f"Cannot build {self} node with input dtype {idtype} "
                    f"and acc_dtype {acc_dtype}, as precision would be lost. "
                    "To correct this error, you can:\n"
                    "  - not specify acc_dtype, or\n"
                    f"  - use an acc_dtype at least as precise as {upcasted_dtype}.\n"
                    '  - specify "dtype" instead of "acc_dtype", so '
                    "the reduction will be precise, but the result will "
                    'be casted into "dtype" at the end.\n'
                    "If you are expecting the precision loss, you can "
                    f'use tensor.cast(..., dtype="{acc_dtype}"), on your input.'
                )
            return acc_dtype

    def make_node(self, input):
        input = as_tensor_variable(input)
        inp_dims = input.type.ndim
        inp_dtype = input.type.dtype

        # We need to redefine make_node so that, if self.dtype is None,
        # we can infer what dtype should be, and create a node from an Op
        # of the appropriate dtype.
        dtype = self._output_dtype(inp_dtype)
        acc_dtype = self._acc_dtype(inp_dtype)

        assert dtype is not None
        assert acc_dtype is not None

        axis = self.axis

        # scalar inputs are treated as 1D regarding axis in this `Op`
        if axis is not None:
            try:
                axis = np.core.numeric.normalize_axis_tuple(axis, ndim=max(1, inp_dims))
            except np.AxisError:
                raise np.AxisError(axis, ndim=inp_dims)

            out_shape = tuple(
                s for i, s in enumerate(input.type.shape) if i not in axis
            )
        else:
            out_shape = ()

        if (
            (axis is not None and any(a < 0 for a in axis))
            or dtype != self.dtype
            or acc_dtype != self.acc_dtype
        ):
            op = self.clone(axis=axis, dtype=dtype, acc_dtype=acc_dtype)
        else:
            op = self

        output = TensorType(dtype=dtype, shape=out_shape)()

        return Apply(op, [input], [output])

    def clone(
        self,
        axis=None,
        dtype=None,
        acc_dtype=None,
        upcast_discrete_output=None,
        **kwargs,
    ):
        if axis is None:
            axis = self.axis
        if dtype is None:
            dtype = self.dtype
        if acc_dtype is None:
            acc_dtype = self.acc_dtype
        if upcast_discrete_output is None:
            upcast_discrete_output = self.upcast_discrete_output

        res = type(self)(
            self.scalar_op,
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            upcast_discrete_output=None,
            **kwargs,
        )

        return res

    def __str__(self):
        prefix = f"{type(self).__name__}{{{self.scalar_op}}}"
        extra_params = []

        if self.axis is not None:
            axis = ", ".join(str(x) for x in self.axis)
            extra_params.append(f"axis=[{axis}]")

        if self.acc_dtype:
            extra_params.append(f"acc_dtype={self.acc_dtype}")

        extra_params_str = ", ".join(extra_params)

        if extra_params_str:
            return f"{prefix}{{{extra_params_str}}}"
        else:
            return f"{prefix}"

    def perform(self, node, inp, out):
        (input,) = inp
        (output,) = out
        axis = self.axis

        out_dtype = node.outputs[0].type.dtype

        if self.acc_dtype is not None:
            acc_dtype = self.acc_dtype
        else:
            acc_dtype = out_dtype

        # out_dtype = self.dtype if self.dtype and self.dtype != "OLD" else out_dtype

        input = np.array(input, dtype=acc_dtype)

        out = self.ufunc.reduce(input, axis=axis, dtype=acc_dtype)

        output[0] = _asarray(out, dtype=out_dtype)

    def infer_shape(self, fgraph, node, shapes):
        (ishape,) = shapes
        axis = self.axis
        if axis is None:
            return ((),)
        return ([ishape[i] for i in range(node.inputs[0].type.ndim) if i not in axis],)

    def _c_all(self, node, name, inames, onames, sub):

        input = node.inputs[0]
        output = node.outputs[0]

        iname = inames[0]
        oname = onames[0]

        idtype = input.type.dtype_specs()[1]
        odtype = output.type.dtype_specs()[1]

        acc_dtype = getattr(self, "acc_dtype", None)

        if acc_dtype is not None:
            if acc_dtype == "float16":
                raise MethodNotDefined("no c_code for float16")
            acc_type = TensorType(shape=node.outputs[0].type.shape, dtype=acc_dtype)
            adtype = acc_type.dtype_specs()[1]
        else:
            adtype = odtype

        axis = self.axis
        if axis is None:
            axis = list(range(input.type.ndim))

        if len(axis) == 0:
            # The acc_dtype is never a downcast compared to the input dtype
            # So we just need a cast to the output dtype.
            var = pytensor.tensor.basic.cast(input, node.outputs[0].dtype)
            if var is input:
                var = Elemwise(scalar_identity)(input)
            assert var.dtype == node.outputs[0].dtype
            return var.owner.op._c_all(var.owner, name, inames, onames, sub)

        order1 = [i for i in range(input.type.ndim) if i not in axis]
        order = order1 + list(axis)

        nnested = len(order1)

        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(node.inputs, inames)):
            sub[f"lv{i}"] = iname

        decl = ""
        if adtype != odtype:
            # Create an accumulator variable different from the output
            aname = "acc"
            decl = acc_type.c_declare(aname, sub)
            decl += acc_type.c_init(aname, sub)
        else:
            # the output is the accumulator variable
            aname = oname

        decl += cgen.make_declare([order], [idtype], sub)
        checks = cgen.make_checks([order], [idtype], sub)

        alloc = ""
        i += 1
        sub[f"lv{i}"] = oname
        sub["olv"] = oname

        # Allocate output buffer
        alloc += cgen.make_declare(
            [list(range(nnested)) + ["x"] * len(axis)], [odtype], dict(sub, lv0=oname)
        )
        alloc += cgen.make_alloc([order1], odtype, sub)
        alloc += cgen.make_checks(
            [list(range(nnested)) + ["x"] * len(axis)], [odtype], dict(sub, lv0=oname)
        )

        if adtype != odtype:
            # Allocate accumulation buffer
            sub[f"lv{i}"] = aname
            sub["olv"] = aname

            alloc += cgen.make_declare(
                [list(range(nnested)) + ["x"] * len(axis)],
                [adtype],
                dict(sub, lv0=aname),
            )
            alloc += cgen.make_alloc([order1], adtype, sub)
            alloc += cgen.make_checks(
                [list(range(nnested)) + ["x"] * len(axis)],
                [adtype],
                dict(sub, lv0=aname),
            )

        identity = self.scalar_op.identity

        if np.isposinf(identity):
            if input.type.dtype in ("float32", "float64"):
                identity = "__builtin_inf()"
            elif input.type.dtype.startswith("uint") or input.type.dtype == "bool":
                identity = "1"
            else:
                identity = "NPY_MAX_" + str(input.type.dtype).upper()
        elif np.isneginf(identity):
            if input.type.dtype in ("float32", "float64"):
                identity = "-__builtin_inf()"
            elif input.type.dtype.startswith("uint") or input.type.dtype == "bool":
                identity = "0"
            else:
                identity = "NPY_MIN_" + str(input.type.dtype).upper()
        elif identity is None:
            raise TypeError(f"The {self.scalar_op} does not define an identity.")

        task0_decl = (
            f"{adtype}& {aname}_i = *{aname}_iter;\n" f"{aname}_i = {identity};"
        )

        task1_decl = f"{idtype}& {inames[0]}_i = *{inames[0]}_iter;\n"

        task1_code = self.scalar_op.c_code(
            Apply(
                self.scalar_op,
                [
                    get_scalar_type(dtype=iv.type.dtype).make_variable()
                    for iv in (node.inputs * 2)
                ],
                [
                    get_scalar_type(dtype=ov.type.dtype).make_variable()
                    for ov in node.outputs
                ],
            ),
            None,
            [f"{aname}_i", f"{inames[0]}_i"],
            [f"{aname}_i"],
            sub,
        )
        code1 = f"""
        {{
            {task1_decl}
            {task1_code}
        }}
        """

        if node.inputs[0].type.ndim:
            if len(axis) == 1:
                all_code = [("", "")] * nnested + [(task0_decl, code1), ""]
            else:
                all_code = (
                    [("", "")] * nnested
                    + [(task0_decl, "")]
                    + [("", "")] * (len(axis) - 2)
                    + [("", code1), ""]
                )
        else:
            all_code = [task0_decl + code1]
        loop = cgen.make_loop_careduce(
            [order, list(range(nnested)) + ["x"] * len(axis)],
            [idtype, adtype],
            all_code,
            sub,
        )

        end = ""
        if adtype != odtype:
            end = f"""
            PyArray_CopyInto({oname}, {aname});
            """
            end += acc_type.c_cleanup(aname, sub)

        return decl, checks, alloc, loop, end

    def c_code(self, node, name, inames, onames, sub):
        code = "\n".join(self._c_all(node, name, inames, onames, sub))
        return code

    def c_headers(self, **kwargs):
        # Sometimes, Elemwise's c_code is returned, so we need its headers
        return ["<vector>", "<algorithm>"]

    def c_code_cache_version_apply(self, node):
        # the version corresponding to the c code in this Op
        version = [9]

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(
            self.scalar_op,
            [
                get_scalar_type(dtype=input.type.dtype).make_variable()
                for input in node.inputs
            ],
            [
                get_scalar_type(dtype=output.type.dtype).make_variable()
                for output in node.outputs
            ],
        )
        version.append(self.scalar_op.c_code_cache_version_apply(scalar_node))
        for i in node.inputs + node.outputs:
            version.append(get_scalar_type(dtype=i.type.dtype).c_code_cache_version())
        if all(version):
            return tuple(version)
        else:
            return ()


def scalar_elemwise(*symbol, nfunc=None, nin=None, nout=None, symbolname=None):
    """Replace a symbol definition with an `Elemwise`-wrapped version of the corresponding scalar `Op`.

    If it is not ``None``, the `nfunc` argument should be a string such that
    ``getattr(numpy, nfunc)`` implements a vectorized version of the `Elemwise`
    operation.  `nin` is the number of inputs expected by that function, and nout
    is the number of **destination** inputs it takes.  That is, the function
    should take nin + nout inputs. `nout == 0` means that the numpy function does
    not take a NumPy array argument to put its result in.

    """
    import pytensor.scalar as scalar

    def construct(symbol):
        nonlocal symbolname

        symbolname = symbolname or symbol.__name__

        if symbolname.endswith("_inplace"):
            elemwise_name = f"Elemwise{{{symbolname},inplace}}"
            scalar_op = getattr(scalar, symbolname[: -len("_inplace")])
            inplace_scalar_op = scalar_op.__class__(transfer_type(0))
            rval = Elemwise(
                inplace_scalar_op,
                {0: 0},
                name=elemwise_name,
                nfunc_spec=(nfunc and (nfunc, nin, nout)),
            )
        else:
            elemwise_name = f"Elemwise{{{symbolname},no_inplace}}"
            scalar_op = getattr(scalar, symbolname)
            rval = Elemwise(
                scalar_op, name=elemwise_name, nfunc_spec=(nfunc and (nfunc, nin, nout))
            )

        if getattr(symbol, "__doc__"):
            rval.__doc__ = symbol.__doc__ + "\n" + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = symbol.__module__

        pprint.assign(rval, FunctionPrinter([symbolname.replace("_inplace", "=")]))

        return rval

    if symbol:
        return construct(symbol[0])
    else:
        return construct


@_get_vector_length.register(Elemwise)
def _get_vector_length_Elemwise(op, var):
    if len(var.owner.inputs) == 1 and len(var.owner.outputs) == 1:
        return get_vector_length(var.owner.inputs[0])

    raise ValueError(f"Length of {var} cannot be determined")
