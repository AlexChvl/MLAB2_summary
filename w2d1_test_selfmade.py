import torch as t
import torch.nn as nn

from utils import allclose, allclose_atol, assert_all_equal, report



@report
def test_layer_norm(LayerNorm):
    x = t.rand((8, 9, 10))
    ln1= nn.LayerNorm((9, 10), 1e-05, elementwise_affine=False)
    ln = LayerNorm((9, 10), elementwise_affine = False)
    actual = ln.forward(x)
    expected = ln1.forward(x)
    allclose(actual, expected)

    ln2 = nn.LayerNorm((9, 10), 1e-05, elementwise_affine=True)
    ln = LayerNorm((9, 10), elementwise_affine = True)
    assert_all_equal(ln.weights, ln2.weight)
    assert_all_equal(ln.biases, ln2.bias)
    actual = ln.forward(x)
    expected = ln2.forward(x)
    allclose(actual, expected)
    

 