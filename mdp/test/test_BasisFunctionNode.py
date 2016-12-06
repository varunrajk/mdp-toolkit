
from ._tools import *
from mdp.nodes import BasisFunctionNode
import mdp

# TODO Need to add more test functions.


def test_basisfn_node():

    # test indicator fn
    lims = [[0], [10]]
    bfn = BasisFunctionNode(basis_name='indicator', lims=lims)

    for i in xrange(lims[1][0]):
        inp = mdp.numx.array([[i]])
        out = bfn(inp)
        exp_out = mdp.numx.zeros(lims[1][0] - lims[0][0] + 1)[None, :]
        exp_out[0, i] = 1.
        assert_array_equal(out, exp_out)
