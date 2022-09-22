from typing import Callable
import torch as t
from torch import nn
from utils import report, report_success, allclose, assert_all_equal, allclose_atol
import einops
from einops import rearrange, repeat, reduce

@report
def test_einsum_trace(trace_fn):
    assert trace_fn(t.ones(5,5)) == 5
    assert trace_fn(t.as_strided(t.Tensor([0]), (5, 5), (0, 0))) == 0


@report 
def test_matmul(mul_fn):
    M1 = t.rand(5, 3)
    M2 = t.rand(3, 4)
    assert_all_equal( mul_fn(M1, M2), M1@M2)

    M1= rearrange(t.arange(6), "(a b)-> a b", a = 2)
    M2 = rearrange(t.arange(6), "(a b) -> a b", a = 3)
    assert_all_equal(mul_fn(M1, M2), M1@M2)


@report
def test_conv1d_minimal(conv_fn):
    m = nn.Conv1d(1, 1, 2, bias = False)
    x = t.arange(25).float().view(1, 1, 25)
    # x.shape = (1, 1, 25)
    with t.inference_mode():
        y = m(x)
    actual = conv_fn(x, m.weight)
    assert_all_equal(actual, y)


@report
def test_conv2d_minimal(conv_fn):
    m = nn.Conv2d(1, 1, (2,2), bias = False)
    x = t.arange(64).float().view(1, 1, 8, 8)
    with t.inference_mode():
        y = m(x)
    print(y.shape)
    actual = conv_fn(x, m.weight)
    print(actual.shape)
    allclose(actual, y)


@report 
def test_pad1d(pad_fn):
    x = t.rand(2, 3, 5)
    left = repeat(t.ones(1), "1 -> 2 3 2")
    right = repeat(t.ones(1), "1 -> 2 3 3")
    correct = t.cat((left, x, right), dim = -1)
    assert_all_equal(pad_fn(x, 2, 3, 1.0), correct)


@report 
def test_pad2d(pad_fn):
    #add 1 on left and right and 2 top and bottom
    x = t.rand(2, 3, 5, 6)
    left = repeat(t.ones(1), "1 -> 2 3 5 1")
    right = repeat(t.ones(1), "1 -> 2 3 5 1")

    y = t.cat((left, x, right), dim = -1)

    top = repeat(t.ones(1), "1 -> 2 3 2 8")
    bottom = repeat(t.ones(1), "1 -> 2 3 2 8")

    correct = t.cat((top, y, bottom), dim = -2)
    assert_all_equal(correct, pad_fn(x, 1, 1, 2, 2, 1.0))



