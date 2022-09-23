#%%

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Union
import numpy as np
import einops

from matplotlib import pyplot as plt

# import w1d3_test
import w1d3_utils

import w1d3_test_selfmade


MAIN = __name__ == "__main__"
Arr = np.ndarray
grad_tracking_enabled = True


@dataclass
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    """The 'inner' NumPy function that does the actual forward computation."""

    args: tuple[Any]
    """The unwrapped input arguments, meaning raw NumPy arrays and not Tensors where applicable."""

    kwargs: dict[str, Any]
    """Keyword arguments. To keep things simple today, we aren't going to backpropagate with respect to these."""

    parents: dict[int, "Tensor"]
    """Map from positional argument index to the Tensor at that position."""


class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.

    There is a lot of repetitive boilerplate involved which we have done for you.
    You don't need to modify anything in this class: the methods here will delegate to functions that you will implement throughout the day.
    """

    array: Arr
    """The underlying array. Can be shared between multiple Tensors."""

    requires_grad: bool
    """If True, calling functions or methods on this tensor will track relevant data for backprop."""

    grad: Optional["Tensor"]
    """Backpropagation will accumulate gradients into this field."""

    recipe: Optional[Recipe]
    """Extra information necessary to run backpropagation."""

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None

        self.recipe = None
        """If not None, this tensor was created via recipe.func(*recipe.args, **recipe.kwargs)."""

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad:
            if self.recipe is None:
                return True
            if self.recipe.parents:
                return False
            return True
        return True


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


# %%


def log_forward(x: Tensor) -> Tensor:
    """recreate forward pass of log function"""
    out = Tensor(np.log(x.array), (grad_tracking_enabled and (x.requires_grad)))

    if grad_tracking_enabled and (x.requires_grad or x.recipe):
        rec = Recipe(np.log, (x.array,), {}, {0: x})
    else:
        rec = None
    out.recipe = rec

    return out


if MAIN:
    log = log_forward
    w1d3_test_selfmade.test_log(Tensor, log_forward)

# %%


def topological_sort(node: Tensor) -> list[Tensor]:
    """
    Return a list of node's descendants in reverse topological order from future to past.

    Use the depth-first search from [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting)
    """
    ordering = []
    temp = []

    def visit(v: Tensor) -> None:
        temp.append(v)
        if v.recipe is None:
            ordering.append(v)
            return
        for u in v.recipe.parents.values():
            if (u not in set(ordering)) and (u not in set(temp)):
                visit(u)
        ordering.append(v)

    visit(node)
    return ordering[::-1]


if MAIN:
    w1d3_test_selfmade.test_topological_sort(topological_sort)


# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x). Provided as an optimization in case it's cheaper to express the gradient in terms of the output.
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x


# %%
BACK_FUNCS: defaultdict[Callable, dict[int, Callable]] = defaultdict(dict)
BACK_FUNCS[np.log][0] = log_back

# %%
def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    """Accumulate gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: the rightmost node in the computation graph
    end_grad: the grad of the loss wrt end_node: all 1s if not specified.
    """

    ordering = topological_sort(end_node)
    tmps = {node: np.zeros_like(node.array, float) for node in ordering}  # dict of gradients associated to each node

    if end_grad is None:
        end_grad = np.ones(end_node.array.shape)
    else:
        end_grad = end_grad.array
    tmps[end_node] = end_grad

    for node in ordering:
        if node.recipe is not None:
            back_funcs = BACK_FUNCS[node.recipe.func]

            for idx, parent_node in node.recipe.parents.items():
                tmps[parent_node] += back_funcs[idx](
                    tmps[node],
                    node.array,
                    *node.recipe.args,
                    **node.recipe.kwargs,
                )

    for node in ordering:
        if (node.recipe is None) and (node.requires_grad):
            if node.grad is None:
                node.grad = Tensor(tmps[node], True)
            else:
                node.grad.array += tmps[node]


if MAIN:
    w1d3_test_selfmade.test_backprop(backprop)


# %%


def wrap(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: function. It takes any number of positional arguments, some of which may be NumPy arrays, and any number of keyword arguments which we aren't allowing to be NumPy arrays at present. It returns a single NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input argument, so we may need to track information in a Recipe. If False, we definitely don't need to track information.

    Return: function. It has the same signature as numpy_func, except wherever there was a NumPy array, this has a Tensor instead.
    """
    global grad_tracking_enabled

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        arr_args = [arg.array if isinstance(arg, Tensor) else arg for arg in args]
        # print("arguments", arr_args)
        output = numpy_func(*arr_args, **kwargs)

        requires_grad = (
            any([arg.requires_grad for arg in args if isinstance(arg, Tensor)])
            and grad_tracking_enabled
            and is_differentiable
        )

        sol = Tensor(output, requires_grad)

        tensor_args = [arg for arg in args if isinstance(arg, Tensor)]
        parents = {i: arg for i, arg in enumerate(tensor_args) if (arg.requires_grad or arg.recipe is not None)}

        if requires_grad and is_differentiable:
            sol.recipe = Recipe(numpy_func, arr_args, kwargs, parents)
        return sol

    return tensor_func


log = wrap(np.log)
if MAIN:
    w1d3_test_selfmade.test_log(Tensor, log)
    try:
        log(x=Tensor([100]))
    except Exception as e:
        print("Got a nice exception as intended:")
        print(e)
    else:
        assert False, "Passing tensor by keyword should raise some informative exception."

# %%
def _argmax(x: Arr, dim=None, keepdim=False):
    """Like torch.argmax."""
    return np.argmax(x, axis=dim, keepdims=keepdim)


argmax = wrap(_argmax, is_differentiable=False)
eq = wrap(np.equal, is_differentiable=False)


if MAIN:
    a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
    b = a.argmax()
    assert not b.requires_grad
    assert b.recipe is None
    assert b.item() == 3
# %%


def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return -np.ones(grad_out.shape) * grad_out


negative = wrap(np.negative)
BACK_FUNCS[np.negative][0] = negative_back


def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out * out


exp = wrap(np.exp)
BACK_FUNCS[np.exp][0] = exp_back


def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)


reshape = wrap(np.reshape)
BACK_FUNCS[np.reshape][0] = reshape_back


def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    new = np.argsort(axes)
    return np.transpose(grad_out, new)


BACK_FUNCS[np.transpose][0] = permute_back
permute = wrap(np.transpose)


def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    """
    original_shape = original.shape
    extra_dims = len(broadcasted.shape) - len(original_shape)

    expanded = [dim for dim in range(len(original_shape)) if original_shape[dim] != broadcasted.shape[dim + extra_dims]]
    broadcasted = np.sum(broadcasted, axis=tuple(range(extra_dims)))

    broadcasted = np.sum(broadcasted, tuple(expanded), keepdims=True)
    return broadcasted


def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)


def _expand(x: Arr, new_shape) -> Arr:
    """Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    new_dims = len(new_shape) - len(x.shape)
    old = [0] * new_dims + list(x.shape)
    new = []
    for ind, s in enumerate(new_shape):
        if s == -1:
            new.append(old[ind])
        else:
            new.append(s)
    new = tuple(new)
    return np.broadcast_to(x, new)


expand = wrap(_expand)
BACK_FUNCS[_expand][0] = expand_back


def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    if not keepdim and dim:
        grad_out = np.expand_dims(grad_out, dim)
    grad_out = np.broadcast_to(grad_out, x.shape)
    return grad_out


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)


sum = wrap(_sum)
BACK_FUNCS[_sum][0] = sum_back
# if MAIN:
#     w1d3_test.test_sum_keepdim_false(Tensor)
#     w1d3_test.test_sum_keepdim_true(Tensor)
#     w1d3_test.test_sum_dim_none(Tensor)

# %%
Index = Union[int, tuple[int], tuple[Arr], tuple[Tensor]]


def _getitem(x: Arr, index: Index) -> Arr:
    """Like x[index] when x is a torch.Tensor."""
    if isinstance(index, tuple) and isinstance(index[0], Tensor):
        # print("index:", index)
        ind = tuple([tens.array for tens in index])
        # print("ind:", ind)
    else:
        ind = index
    return x[ind]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    """Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    """
    z = np.zeros(x.shape)
    if isinstance(index, tuple) and isinstance(index[0], Tensor):
        ind = tuple([tens.array for tens in index])
    else:
        ind = index
    np.add.at(z, ind, grad_out)
    return z


getitem = wrap(_getitem)
BACK_FUNCS[_getitem][0] = getitem_back
# if MAIN:
#     w1d3_test.test_getitem_int(Tensor)
#     w1d3_test.test_getitem_tuple(Tensor)
#     w1d3_test.test_getitem_integer_array(Tensor)
#     w1d3_test.test_getitem_integer_tensor(Tensor)


# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    return unbroadcast(grad_out * y, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    return unbroadcast(grad_out * x, y)


multiply = wrap(np.multiply)
BACK_FUNCS[np.multiply][0] = multiply_back0
BACK_FUNCS[np.multiply][1] = multiply_back1
# if MAIN:
#     w1d3_test.test_multiply_broadcasted(Tensor)

# %%
add = wrap(np.add)
subtract = wrap(np.subtract)
true_divide = wrap(np.true_divide)
BACK_FUNCS[np.add][0] = lambda grad_out, out, x, y: unbroadcast(grad_out, x)
BACK_FUNCS[np.add][1] = lambda grad_out, out, x, y: unbroadcast(grad_out, y)
BACK_FUNCS[np.subtract][0] = lambda grad_out, out, x, y: unbroadcast(grad_out, x)
BACK_FUNCS[np.subtract][1] = lambda grad_out, out, x, y: unbroadcast(-grad_out, y)
BACK_FUNCS[np.true_divide][0] = lambda grad_out, out, x, y: unbroadcast(grad_out / y, x)
BACK_FUNCS[np.true_divide][1] = lambda grad_out, out, x, y: unbroadcast(-(grad_out * x) / (y**2), y)

# if MAIN:
#     w1d3_test.test_add_broadcasted(Tensor)
#     w1d3_test.test_subtract_broadcasted(Tensor)
#     w1d3_test.test_truedivide_broadcasted(Tensor)

# %%
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    """Like torch.add_. Compute x += other * alpha in-place and return tensor."""
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    """This example should work properly."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    """This example is expected to compute the wrong gradients."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")


if MAIN:
    safe_example()
    unsafe_example()


# %%


def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt x."""
    return unbroadcast(grad_out * (x > y), x) + unbroadcast(grad_out * (x == y), x) / 2


def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt y."""
    return unbroadcast(grad_out * (y > x), y) + unbroadcast(grad_out * (x == y), y) / 2


maximum = wrap(np.maximum)
BACK_FUNCS[np.maximum][0] = maximum_back0
BACK_FUNCS[np.maximum][1] = maximum_back1
# if MAIN:
#     w1d3_test.test_maximum(Tensor)
#     w1d3_test.test_maximum_broadcasted(Tensor)


# %%


def relu(x: Tensor) -> Tensor:
    """Like torch.nn.function.relu(x, inplace=False)."""
    return maximum(x, Tensor(np.zeros_like(x.array), requires_grad=False))


# if MAIN:
#     w1d3_test.test_relu(Tensor)


#%%


def _matmul2d(x: Arr, y: Arr) -> Arr:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y


def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    """backprop for out = x*y along x where both matrices are 2 dimensional."""
    assert len(grad_out.shape) == 2
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    return grad_out @ y.T


def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    """backprop for out = x*y along y where both matrices are 2 dimensional."""
    assert len(grad_out.shape) == 2
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    return x.T @ grad_out


matmul = wrap(_matmul2d)
BACK_FUNCS[_matmul2d][0] = matmul2d_back0
BACK_FUNCS[_matmul2d][1] = matmul2d_back1
# if MAIN:
#     w1d3_test.test_matmul2d(Tensor)
# %%


class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        """Share the array with the provided tensor."""
        super().__init__(tensor.array, requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"


if MAIN:
    x = Tensor([1.0, 2.0, 3.0])
    p = Parameter(x)
    assert p.requires_grad
    assert p.array is x.array
    assert repr(p) == "Parameter containing:\nTensor(array([1., 2., 3.]), requires_grad=True)"
    x.add_(Tensor(np.array(2.0)))
    assert np.allclose(
        p.array, np.array([3.0, 4.0, 5.0])
    ), "in-place modifications to the original tensor should affect the parameter"


# %%
class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        paramlist = []
        paramlist += [value for value in self._parameters.values()]
        if recurse:
            for module in self.modules():
                paramlist += module.parameters(recurse=True)
        return paramlist

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call the superclass.
        """
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """
        if key in self._parameters.keys():
            return self._parameters[key]
        elif key in self._modules.keys():
            return self._modules[key]
        else:
            raise KeyError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        return f"{self._modules=},{self._parameters=}"


if MAIN:

    class TestInnerModule(Module):
        def __init__(self):
            super().__init__()
            self.param1 = Parameter(Tensor([1.0]))
            self.param2 = Parameter(Tensor([2.0]))

    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.inner = TestInnerModule()
            self.param3 = Parameter(Tensor([3.0]))

    mod = TestModule()
    assert list(mod.modules()) == [mod.inner]
    print(list(mod.parameters()))
    print(
        [
            mod.param3,
            mod.inner.param1,
            mod.inner.param2,
        ]
    )
    assert list(mod.parameters()) == [
        mod.param3,
        mod.inner.param1,
        mod.inner.param2,
    ], "parameters should come before submodule parameters"
    print("Manually verify that the repr looks reasonable:")
    print(mod)


# %%


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        bound = in_features**-0.5
        self.weight = Parameter(empty(out_features, in_features).uniform_(-bound, bound))
        bias_param = Parameter(empty(out_features).uniform_(-bound, bound))

        if bias:
            self.bias = Parameter(bias_param)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        if self.bias is not None:
            return x @ permute(self.weight, (1, 0)) + self.bias
        else:
            return x @ permute(self.weight, (1, 0))

    def extra_repr(self) -> str:
        return f"{self.in_features=}, {self.out_features=}, {self.weight=}, {self.bias=}"


# %%


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 10)

    def forward(self, x):
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.linear1(x)
        x = relu(x)
        return x


# %%
def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,)

    Return: shape (batch, ) containing the per-example loss.
    """
    max_ind = argmax(logits, dim=1)
    max_val = logits[arange(0, logits.shape[0], 1), max_ind]

    max_val = permute(expand(max_val, logits.shape[::-1]), (1, 0))  # shape (classes, batch)
    logits -= max_val  # shift for numerical stability

    exped = exp(logits)
    denom = exped.sum(dim=1, keepdim=True)
    prob = exped / denom
    return -log(prob[arange(0, logits.shape[0], 1), true_labels])


# if MAIN:
#     w1d3_test.test_cross_entropy(Tensor, cross_entropy)
# %%


class NoGrad:
    """Context manager that disables grad inside the block. Like torch.no_grad."""

    was_enabled: bool

    def __enter__(self):
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        global grad_tracking_enabled

        grad_tracking_enabled = self.was_enabled


# if MAIN:
#     w1d3_test.test_no_grad(Tensor, NoGrad)
#     w1d3_test.test_no_grad_nested(Tensor, NoGrad)
# %%


def visualize(dataloader):
    """Call this if you want to see some of your data."""
    plt.figure(figsize=(12, 12))
    (sample, sample_labels) = next(iter(dataloader))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(sample[i, 0], cmap=plt.cm.binary)
    plt.show()


if MAIN:
    (train_loader, test_loader) = w1d3_utils.get_mnist()

#%%
class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        """Vanilla SGD with no additional features."""
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with NoGrad():
            for param in self.params:
                param.add_(param.grad, -self.lr)


#%%
def train(model, train_loader, optimizer, epoch):
    for (batch_idx, (data, target)) in enumerate(train_loader):
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)

        loss = sum(cross_entropy(output, target)) / len(output)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with NoGrad():
        for (data, target) in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


# %%
if MAIN:
    num_epochs = 50
    model = MLP()
    start = time.time()
    optimizer = SGD(model.parameters(), 0.01)
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        optimizer.step()
    print(f"Completed in {time.time() - start: .2f}s")
# %%
