import numpy as np

from utils import report, allclose, assert_all_equal
from typing import Callable
import warnings
from w1d3_answers import Recipe, Tensor, log_forward


@report
def test_log(Tensor, log_fwd):
    a = Tensor([np.e, 1.0], requires_grad=True)
    b = log_fwd(a)

    assert np.allclose(b.array, np.array([1.0, 0.0])), f"wrong array:{b.array}"
    assert b.requires_grad, "wrong gradient"
    assert b.recipe is not None, "No recipe?"
    assert b.is_leaf == False, "recipe didn't update correctly. b is a leaf?"
    assert b.recipe.func == np.log, "wrong function in recipe"
    assert b.recipe.kwargs == {}, "found kwargs?"
    assert len(b.recipe.args) == 1 and np.allclose(
        b.recipe.args[0], a.array
    ), f"args were not the tuple input? {b.recipe.args=}"
    # assert b.recipe.args == (a.array,), f"args were not the tuple input? {b.recipe.args=}"
    assert b.recipe.parents == {0: a}, "wrong parents in recipe"


@report
def test_topological_sort(top_sort):
    """basic test, not exhaustive."""
    a = Tensor(np.array([0]), requires_grad=True)
    b = Tensor(np.array([0]), requires_grad=True)
    crec = Recipe(np.sum, (a.array, b.array), {}, {0: a, 1: b})
    c = Tensor(np.array(0), requires_grad=True)
    c.recipe = crec
    actual = top_sort(c)
    expected = [c, b, a]
    assert actual == expected


from w1d3_answers import log_back


@report
def test_backprop(backprop):
    a = Tensor(np.array([2 * np.e, 4 * np.e]), requires_grad=True)
    b = log_forward(a)
    c = log_forward(b)
    c.backward()
    assert np.allclose(a.grad.array, 1 / (a.array * np.log(a.array))), "failed basic test on log(log(x))"
