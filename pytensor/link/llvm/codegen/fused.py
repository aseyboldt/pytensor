"""FusedElemwise: one loop for an Elemwise with indexed reads, indexed writes and
reductions (see pytensor.tensor.rewriting.fused_elemwise).

Outer inputs are `[elemwise inputs..., index arrays..., write targets...]`. Each Elemwise
input is plain (broadcast by its type) or read through integer index arrays on consecutive
axes, which broadcast together (as in numpy) over loop axes of their own. Each output is
plain, a reduction over some loop axes, or written through such indices into a copy of
its target (inc or set). Patterns outside this raise NotImplementedError and
the planner runs the inner graph node by node instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.codegen.elemwise import identity_value
from pytensor.link.llvm.codegen.emit import (
    Array,
    ProvidedOutput,
    ScalarOutput,
    advance,
    cast,
    clamped_index,
    constant,
    copy,
    for_each,
    index_bounds,
    is_contiguous,
    load,
    loop_order,
    loop_view,
    memory_type,
    size,
    store,
)
from pytensor.link.llvm.codegen.function import llvm_emit
from pytensor.link.llvm.codegen.scalar import emit_scalar
from pytensor.link.llvm.codegen.types import check_dtype, ir
from pytensor.link.llvm.layout import DYNAMIC, Layout
from pytensor.scalar.basic import ScalarType, add
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.subtensor import AdvancedSubtensor


#: elements of the last axis whose values to write are computed, then written, at once
CHUNK = 128


@dataclass
class Read:
    index: int  # outer input position of the index array
    axis: int


@dataclass
class Write:
    reads: list  # [Read]: the index arrays and the target axes they index
    target: int  # outer input position of the target
    set: bool


@dataclass
class Reduce:
    scalar_op: object
    axes: tuple
    identity: object
    acc: str


def fused_spec(op, node):
    """(the Elemwise, its number of inputs, per input its `Read`s or None, per output
    its `Reduce`/`Write` or None, the loop axes of each index array)."""
    inner = op.fgraph
    elems = [n for n in inner.apply_nodes if isinstance(n.op, Elemwise)]
    if len(elems) != 1:
        raise NotImplementedError("FusedElemwise: expected one inner Elemwise")
    (E,) = elems
    n_idx = len(op.indexed_inputs)
    if len(op.indexed_outputs) not in (0, n_idx):
        raise NotImplementedError("FusedElemwise: inconsistent index specs")
    written = sorted(
        {int(s) for e in op.indexed_outputs if e is not None for s in e[0]}
    )
    n_elem = len(node.inputs) - n_idx - len(written)
    if len(E.inputs) != n_elem or len(E.outputs) != len(node.outputs):
        raise NotImplementedError("FusedElemwise: unexpected inner structure")
    for k in range(n_idx):
        t = node.inputs[n_elem + k].type
        if not t.dtype.startswith(("int", "uint")):
            raise NotImplementedError(
                "FusedElemwise: only integer index arrays are fused"
            )

    reads = [None] * n_elem
    for k, entry in enumerate(op.indexed_inputs):
        if entry is not None:
            for s in entry[0]:
                reads[s] = [*(reads[s] or []), Read(n_elem + k, int(entry[1]))]

    def is_input(v, j):
        # inner inputs fed by one outer variable are merged (a table read through several indices)
        return any(
            v is iv and node.inputs[k] is node.inputs[j]
            for k, iv in enumerate(inner.inputs)
        )

    for j, e_in in enumerate(E.inputs):
        if reads[j] is not None:
            if not (
                e_in.owner
                and isinstance(e_in.owner.op, AdvancedSubtensor)
                and is_input(e_in.owner.inputs[0], j)
            ):
                raise NotImplementedError("FusedElemwise: unexpected indexed read")
        elif not is_input(e_in, j) and not (
            isinstance(node.inputs[j].type, ScalarType) and all(e_in.type.broadcastable)
        ):
            raise NotImplementedError("FusedElemwise: transformed Elemwise input")
    outs = [None] * len(node.outputs)
    for i, spec in enumerate(op.reduced_outputs):
        if spec is None:
            continue
        scalar_op, axes, identity, acc = spec
        if node.outputs[i].type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError("FusedElemwise: reduction changes the dtype")
        outs[i] = Reduce(
            scalar_op, tuple(int(a) for a in axes), identity, check_dtype(acc)
        )
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, axis, mode = entry
        for s in sources:
            if isinstance(outs[s], Reduce):
                raise NotImplementedError(
                    "FusedElemwise: a reduction written through an index"
                )
            if outs[s] is None:
                target = n_elem + n_idx + written.index(int(s))
                outs[s] = Write([], target, mode == "set")
            elif outs[s].set != (mode == "set"):
                raise NotImplementedError("FusedElemwise: mixed inc and set")
            outs[s].reads.append(Read(n_elem + k, int(axis)))
    for i, o in enumerate(node.outputs):
        if o.type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError(
                "FusedElemwise: output dtype differs from the loop's"
            )
    groups = [r for r in reads if r] + [o.reads for o in outs if isinstance(o, Write)]
    for g in groups:
        axes = sorted(r.axis for r in g)
        if axes != list(range(axes[0], axes[0] + len(g))):
            raise NotImplementedError("FusedElemwise: indices on non-consecutive axes")
    idx_axes = index_axes(node, n_elem, n_idx, groups)
    for g in groups:
        end = min(r.axis for r in g) + max(node.inputs[r.index].type.ndim for r in g)
        if any(idx_axes[r.index][-1:] not in ((), (end - 1,)) for r in g):
            raise NotImplementedError(
                "FusedElemwise: an index shared by unlike subscripts"
            )
    return E, n_elem, reads, outs, idx_axes


def index_axes(node, n_elem, n_idx, groups):
    """The loop axes each index array is read along (as numpy broadcasts the indices of
    one subscript: right-aligned over a span starting at the first indexed axis). Index
    arrays shared by several subscripts span the union, as in the numba backend."""
    parent = list(range(n_idx))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for g in groups:
        for r in g[1:]:
            a, c = find(g[0].index - n_elem), find(r.index - n_elem)
            parent[a] = c
    start, width = {}, {}
    for g in groups:
        for r in g:
            k = find(r.index - n_elem)
            start[k] = min(start.get(k, r.axis), r.axis)
            width[k] = max(width.get(k, 0), node.inputs[r.index].type.ndim)
    axes = {}
    for k in range(n_idx):
        root, nd = find(k), node.inputs[n_elem + k].type.ndim
        if root in start:
            end = start[root] + width[root]
            axes[n_elem + k] = tuple(range(end - nd, end))
    return axes


def _register():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    @llvm_emit.register(FusedElemwise)
    def _(op, node, cg, inputs, outputs):
        emit_fused(op, node, cg, inputs, outputs)


def emit_fused(op, node, cg, inputs, outputs):
    b = cg.b
    E, n_elem, reads, outs, idx_axes = fused_spec(op, node)
    nd = E.outputs[0].type.ndim
    in_dtypes = [check_dtype(v.type.dtype) for v in E.inputs]
    out_dtypes = [check_dtype(v.type.dtype) for v in E.outputs]

    def axis_map(ndim, rs):
        """The loop axis of each axis of an array read or written through `rs` (None:
        stepped by its index)."""
        if not rs:
            return list(range(ndim))
        a0, m = min(r.axis for r in rs), len(rs)
        end = a0 + max(node.inputs[r.index].type.ndim for r in rs)
        return [
            d if d < a0 else None if d < a0 + m else d - a0 - m + end
            for d in range(ndim)
        ]

    # ---- the loop shape, from everything that has lengths
    lengths = [[] for _ in range(nd)]  # candidates per axis: (value, what)
    static = list(E.outputs[0].type.shape)

    def index_lengths(rs):
        for r in rs:
            st = node.inputs[r.index].type.shape
            for t, a in enumerate(idx_axes[r.index]):
                if st[t] != 1:
                    lengths[a].append(inputs[r.index].shape[t])

    for j in range(n_elem):
        x = inputs[j]
        if not isinstance(x, Array):
            continue  # a scalar
        etype = E.inputs[j].type
        for d, a in enumerate(axis_map(x.ndim, reads[j])):
            if a is not None and etype.shape[a] != 1:
                lengths[a].append(x.shape[d])
        index_lengths(reads[j] or [])
    for i, spec in enumerate(outs):
        if isinstance(spec, Write):
            t = inputs[spec.target]
            for d, a in enumerate(axis_map(t.ndim, spec.reads)):
                if a is not None:
                    lengths[a].append(t.shape[d])
            index_lengths(spec.reads)
    shape = []
    for a in range(nd):
        cand = lengths[a]
        n = (
            b.i64(static[a])
            if static[a] is not None
            else (cand[0] if cand else b.i64(1))
        )
        for c in cand:
            if not (c.const_int is not None and c.const_int == n.const_int):
                cg.check(b.eq(c, n), f"FusedElemwise: shape mismatch on axis {a}")
        shape.append(n)

    # ---- outputs (ScalarType outputs are computed into a 0-d temporary)
    scalar_outs = {}
    outputs = list(outputs)
    for i, out in enumerate(outputs):
        if isinstance(out, ScalarOutput):
            tmp = cg.temporary(out.dtype, [])
            scalar_outs[i] = (out, tmp)
            outputs[i] = ProvidedOutput(tmp)
    out_arrays = []
    for i, (spec, out) in enumerate(zip(outs, outputs, strict=True)):
        if isinstance(spec, Reduce):
            oshape = [shape[a] for a in range(nd) if a not in spec.axes]
            res = out.allocate(oshape)
            acc = res if spec.acc == res.dtype else cg.temporary(spec.acc, oshape)
            init = constant(
                b, spec.acc, identity_value(spec.scalar_op, spec.identity, spec.acc)
            )
            for_each(
                b,
                oshape,
                [acc],
                lambda p, init=init, spec=spec: store(b, spec.acc, init, p[0]),
            )
            out_arrays.append((res, acc))
        elif isinstance(spec, Write):
            t = inputs[spec.target]
            res = out.allocate(t.shape)
            copy(cg, t, res)
            out_arrays.append((res, None))
        else:
            out_arrays.append((out.allocate(shape), None))

    # ---- operands with loop strides (and their layouts)
    plain_in = {}
    fixed = {}  # inputs that are one element for the whole loop: read once, before it
    for j in range(n_elem):
        x = inputs[j]
        if not isinstance(x, Array):
            continue
        if reads[j] is None and all(s == 1 for s in E.inputs[j].type.shape):
            fixed[j] = load(b, x.dtype, x.data)
            continue
        if reads[j] is None:
            plain_in[j] = loop_view(b, x, shape)
        else:
            plain_in[j] = mapped_view(b, x, shape, axis_map(x.ndim, reads[j]))
    plain_out = {}
    for i, spec in enumerate(outs):
        res, acc = out_arrays[i]
        if isinstance(spec, Reduce):
            plain_out[i] = reduced_view(b, acc, shape, spec.axes)
        elif isinstance(spec, Write):
            plain_out[i] = mapped_view(b, res, shape, axis_map(res.ndim, spec.reads))
        else:
            plain_out[i] = Array(res.dtype, res.data, shape, res.strides)

    def index_value(k, loop_index):
        """The element of index array input k at the loop position (unwrapped)."""
        st = node.inputs[k].type.shape
        at = [
            b.i64(0) if st[t] == 1 else loop_index[a] for t, a in enumerate(idx_axes[k])
        ]
        return inputs[k].load(b, *at).astype("int64").value

    # Index checks don't branch, so the loops vectorize: a bad index is replaced by 0
    # and sets a flag, which fails after the loops; an empty indexed axis fails before.
    flags = {}
    loop_size = size(b, shape)
    for what, arr, rs in [
        *(("read", inputs[j], reads[j]) for j in range(n_elem) if reads[j]),
        *(
            ("write", out_arrays[i][0], s.reads)
            for i, s in enumerate(outs)
            if isinstance(s, Write)
        ),
    ]:
        msg = f"FusedElemwise: index out of bounds ({what})"
        if what not in flags:
            flags[what] = (b.alloca(ir.i8), msg)
            b.store(constant(b, "int8", 0), flags[what][0])
        for r in rs:
            n = arr.shape[r.axis]
            cg.check(b.or_(b.sgt(n, b.i64(0)), b.eq(loop_size, b.i64(0))), msg)

    def checked(v, n, what, k):
        i, ok = clamped_index(b, v, n, index_bounds(node.inputs[k]))
        if ok is not None:
            flag = flags[what][0]
            bad = b.zext(b.not_(ok), ir.i8)
            b.store(b.or_(b.load(ir.i8, flag), bad), flag)
        return i

    def check_flags():
        for flag, msg in flags.values():
            cg.check(b.eq(b.load(ir.i8, flag), constant(b, "int8", 0)), msg)

    def body(ptr_in, ptr_out, loop_index, accs, runs=None, defer=None):
        """One element. `ptr_in[j]`/`ptr_out[i]`: element pointers of the plain parts;
        `accs`: register accumulators {output: value}; `runs`: {output: (index, value)}
        of the incremented element of an output kept in registers (see `inc_run`);
        `defer`: {output: where to store its value instead of writing it through the
        index}. Returns the new accumulators and runs."""
        runs = runs or {}
        new_runs = {}
        args = []
        for j, d in enumerate(in_dtypes):
            x = inputs[j]
            if not isinstance(x, Array):
                args.append((x, d))
                continue
            if j in fixed:
                args.append((cast(b, fixed[j], x.dtype, d), d))
                continue
            p = ptr_in[j]
            for r in reads[j] or []:
                if id(r) in lifted:
                    continue
                ix = checked(
                    index_value(r.index, loop_index), x.shape[r.axis], "read", r.index
                )
                p = advance(b, p, ix, x.strides[r.axis], x.dtype)
            args.append((load(b, x.dtype, p), d))
        res = emit_scalar(E.op.scalar_op, b, args, out_dtypes)
        new_accs = {}
        for i, (spec, v) in enumerate(zip(outs, res, strict=True)):
            d = out_dtypes[i]
            if isinstance(spec, Reduce):
                cur = accs[i] if i in accs else load(b, spec.acc, ptr_out[i])
                new = emit_scalar(
                    spec.scalar_op,
                    b,
                    [(cur, spec.acc), (cast(b, v, d, spec.acc), spec.acc)],
                    [spec.acc],
                )[0]
                if i in accs:
                    new_accs[i] = new
                else:
                    store(b, spec.acc, new, ptr_out[i])
            elif defer and i in defer:
                store(b, d, v, defer[i])
            elif isinstance(spec, Write):
                new_runs[i] = write(i, ptr_out[i], loop_index, v, runs.get(i))
            else:
                store(b, d, v, ptr_out[i])
        return new_accs, new_runs

    def write(i, base, loop_index, v, run=None):
        """`v` written (inc or set) through the indices of output i; the new run if
        `run` (see `inc_run`)."""
        spec, res_arr, d = outs[i], out_arrays[i][0], out_dtypes[i]
        ixs = [
            checked(
                index_value(r.index, loop_index),
                res_arr.shape[r.axis],
                "write",
                r.index,
            )
            for r in spec.reads
            if id(r) not in lifted
        ]
        if run is not None:
            return inc_run(i, d, base, ixs[0], v, *run)
        p = element(i, base, *ixs)
        if not spec.set:
            v = emit_scalar(add, b, [(load(b, d, p), d), (v, d)], [d])[0]
        store(b, d, v, p)

    def element(i, base, *ixs):
        res_arr = out_arrays[i][0]
        rs = [r for r in outs[i].reads if id(r) not in lifted]
        for r, ix in zip(rs, ixs, strict=True):
            base = advance(b, base, ix, res_arr.strides[r.axis], res_arr.dtype)
        return base

    def inc_run(i, d, base, ix, v, prev, cur):
        """`out[ix] += v` with the element being incremented kept in a register while
        the index repeats (`prev`, `cur`: its index, -1 before the first, and value):
        written back when the index changes, so consecutive increments of one element
        don't wait on each other through memory. The order of the additions is kept."""

        def same():
            return [prev, emit_scalar(add, b, [(cur, d), (v, d)], [d])[0]]

        def other():
            with b.if_(b.sge(prev, 0)):
                store(b, d, cur, element(i, base, prev))
            old = load(b, d, element(i, base, ix))
            return [ix, emit_scalar(add, b, [(old, d), (v, d)], [d])[0]]

        return tuple(b.if_else(b.eq(ix, prev), same, other))

    def flush_run(i, d, base, prev, cur):
        with b.if_(b.sge(prev, 0)):
            store(b, d, cur, element(i, base, prev))

    in_keys = sorted(plain_in)
    out_keys = sorted(plain_out)
    arrays = [plain_in[j] for j in in_keys] + [plain_out[i] for i in out_keys]
    any_indexed = any(reads) or any(isinstance(s, Write) for s in outs)

    def split(ptrs):
        return dict(zip(in_keys, ptrs[: len(in_keys)], strict=True)), dict(
            zip(out_keys, ptrs[len(in_keys) :], strict=True)
        )

    # reductions that reduce the innermost axis are carried in registers there
    inner_red = [
        i
        for i, s in enumerate(outs)
        if isinstance(s, Reduce) and nd > 0 and (nd - 1) in s.axes
    ]
    # increments through one index along the innermost axis keep their run in registers
    inner_inc = [
        i
        for i, s in enumerate(outs)
        if isinstance(s, Write)
        and not s.set
        and nd > 0
        and len(s.reads) == 1
        and idx_axes[s.reads[0].index] == (nd - 1,)
    ]

    writes = [i for i in out_keys if isinstance(outs[i], Write)]

    # an index read along outer loop axes only moves its pointer once per iteration of
    # the last of them, not per element: the loop over the last axis stays a stream
    hoisted = {}  # outer axis (-1: before the loops) -> [(array position, Read, array, what)]
    for pos, (rs, arr, what) in enumerate(
        [(reads[j], inputs[j], "read") for j in in_keys]
        + [
            (
                outs[i].reads if i in writes and i not in inner_inc else None,
                out_arrays[i][0],
                "write",
            )
            for i in out_keys
        ]
    ):
        for r in rs or []:
            if nd - 1 not in idx_axes[r.index]:
                hoisted.setdefault(max(idx_axes[r.index], default=-1), []).append(
                    (pos, r, arr, what)
                )
    lifted = {id(r) for rs in hoisted.values() for _, r, _, _ in rs}

    def lift(a, ptrs, index):
        for pos, r, arr, what in hoisted.get(a, []):
            ix = checked(index_value(r.index, index), arr.shape[r.axis], what, r.index)
            ptrs[pos] = advance(b, ptrs[pos], ix, arr.strides[r.axis], arr.dtype)
        return ptrs

    def innermost(a, ptrs, index):
        """The loop over the last axis: reductions over it in registers; values written
        through indices go in chunks, computed by a loop that vectorizes, then written."""
        _, po = split(ptrs)

        def at(i):
            return split(
                [
                    advance(b, p, i, arr.strides[a], arr.dtype)
                    for p, arr in zip(ptrs, arrays, strict=True)
                ]
            )

        def compute(i, accs, defer=None, runs=None):
            qi, qo = at(i)
            new, new_runs = body(
                qi,
                qo,
                [*index, i],
                dict(zip(inner_red, accs, strict=True)),
                runs,
                defer,
            )
            return [new[k] for k in inner_red], new_runs

        def buffer(k, j):
            return b.gep(memory_type(out_dtypes[k]), bufs[k], j)

        def chunk(c, *accs):
            start = b.mul(c, b.i64(CHUNK), nsw=True)
            m = b.smin(b.sub(n, start, nsw=True), b.i64(CHUNK))
            defer = lambda j: {k: buffer(k, j) for k in writes}  # noqa: E731
            accs = b.fold(
                m,
                list(accs),
                lambda j, *accs: compute(b.add(start, j), accs, defer(j))[0],
                vectorize=True,
            )

            # one loop per output: each a single stream of writes, which vectorizes
            for k in writes:

                def scatter(j, *run, k=k):
                    i = b.add(start, j)
                    v = load(b, out_dtypes[k], buffer(k, j))
                    new = write(k, at(i)[1][k], [*index, i], v, run or None)
                    return list(new) if run else []

                run = (
                    [b.i64(-1), constant(b, out_dtypes[k], 0)] if k in inner_inc else []
                )
                run = b.fold(m, run, scatter)
                if k in inner_inc:
                    flush_run(k, out_dtypes[k], po[k], *run)
            return accs

        init = [load(b, outs[i].acc, po[i]) for i in inner_red]
        if writes:
            # outputs of one scalar value (a value written to several targets) share one
            scalar_outs = getattr(E.op.scalar_op, "outputs", None)
            same, bufs = {}, {}
            for k in writes:
                key = (id(scalar_outs[k]) if scalar_outs else k, out_dtypes[k])
                if key not in same:
                    same[key] = b.alloca(memory_type(out_dtypes[k]), CHUNK)
                bufs[k] = same[key]
            n = shape[a]
            n_chunks = b.sdiv(b.add(n, b.i64(CHUNK - 1)), b.i64(CHUNK))
            final = b.fold(n_chunks, init, chunk)
        else:
            final = b.fold(
                shape[a], init, lambda i, *accs: compute(i, accs)[0], vectorize=True
            )
        for i, v in zip(inner_red, final, strict=True):
            store(b, outs[i].acc, v, po[i])

    def nest():
        def level(a, ptrs, index):
            if a == nd - 1 and nd > 0:
                innermost(a, ptrs, index)
                return
            if a == nd:
                pi, po = split(ptrs)
                body(pi, po, index, {})
                return
            with b.loop(shape[a]) as i:
                q = [
                    advance(b, p, i, arr.strides[a], arr.dtype)
                    for p, arr in zip(ptrs, arrays, strict=True)
                ]
                level(a + 1, lift(a, q, [*index, i]), [*index, i])

        level(0, lift(-1, [arr.data for arr in arrays], []), [])
        return []

    # one flat loop when nothing is indexed, every reduction reduces everything, and the
    # plain operands are contiguous (checked at run time unless known)
    flat_ok = (
        not any_indexed
        and nd > 0
        and all(len(s.axes) == nd for s in outs if isinstance(s, Reduce))
        and all(
            E.inputs[j].type.shape[a] != 1 or static[a] == 1
            for j in plain_in
            for a in range(nd)
        )
    )
    if not flat_ok:
        nest()
        check_flags()
        _finish(cg, outs, out_arrays, scalar_outs)
        return

    # every operand packed in the same axis order (from the layouts, so only operands
    # whose layout leaves it open are checked at run time)
    flat_arrays = [plain_in[j] for j in in_keys] + [
        plain_out[i] for i in out_keys if not isinstance(outs[i], Reduce)
    ]
    order = loop_order(shape, flat_arrays)
    contig = b.const_bool(True)
    for arr in flat_arrays:
        contig = b.and_(contig, is_contiguous(b, shape, arr, order))

    def flat():
        red = [i for i in out_keys if isinstance(outs[i], Reduce)]
        init = [load(b, outs[i].acc, plain_out[i].data) for i in red]

        def step(i, *accs):
            pi = {
                j: b.gep(memory_type(plain_in[j].dtype), plain_in[j].data, i)
                for j in in_keys
            }
            po = {
                k: (
                    plain_out[k].data
                    if isinstance(outs[k], Reduce)
                    else b.gep(memory_type(plain_out[k].dtype), plain_out[k].data, i)
                )
                for k in out_keys
            }
            new, _ = body(pi, po, [None] * nd, dict(zip(red, accs, strict=True)))
            return [new[k] for k in red]

        final = b.fold(size(b, shape), init, step, vectorize=True)
        for i, v in zip(red, final, strict=True):
            store(b, outs[i].acc, v, plain_out[i].data)
        return []

    c = contig.const_int
    if c == 1:
        flat()
    elif c == 0:
        nest()
    else:
        b.if_else(contig, flat, nest)
    check_flags()
    _finish(cg, outs, out_arrays, scalar_outs)


def mapped_view(b, a: Array, shape, axes) -> Array:
    """`a` iterated over the loop `shape`, its axis d along loop axis `axes[d]` (None:
    not by the loop); loop axes no axis of `a` (or a broadcast one) runs along get stride 0."""
    nd, static = len(shape), a.static_shape()
    strides, kinds = [b.i64(0)] * nd, [0] * nd
    for d, ax in enumerate(axes):
        if ax is not None and not (static[d] == 1 and shape[ax].const_int != 1):
            strides[ax], kinds[ax] = a.strides[d], DYNAMIC
    return Array(
        a.dtype,
        a.data,
        list(shape),
        strides,
        Layout.make([n.const_int for n in shape], range(nd), kinds),
    )


def reduced_view(b, acc: Array, shape, axes) -> Array:
    """The accumulator of a reduction over `axes` iterated over the full loop `shape`:
    stride 0 on the reduced axes, the accumulator's own strides (and layout) elsewhere."""
    nd = len(shape)
    kept = [a for a in range(nd) if a not in axes]
    strides = [b.i64(0)] * nd
    kinds = [0] * nd
    for k, a in enumerate(kept):
        strides[a] = acc.strides[k]
        kinds[a] = acc.layout.strides[k]
    order = [a for a in range(nd) if a in axes] + [kept[k] for k in acc.layout.order]
    return Array(
        acc.dtype,
        acc.data,
        list(shape),
        strides,
        Layout.make([n.const_int for n in shape], order, kinds),
    )


def _finish(cg, outs, out_arrays, scalar_outs):
    """Convert reductions accumulated in another dtype; set scalar outputs."""
    for i, spec in enumerate(outs):
        if isinstance(spec, Reduce):
            res, acc = out_arrays[i]
            if acc is not res:
                copy(cg, acc, res)
    for out, tmp in scalar_outs.values():
        out.set(cg.b, load(cg.b, tmp.dtype, tmp.data))


_register()

__all__ = ["emit_fused", "fused_spec"]
