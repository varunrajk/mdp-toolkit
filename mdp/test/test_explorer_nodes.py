from ._tools import *
from mdp.nodes import (DiscreteExplorerNode, ContinuousExplorerNode,
                       EpsilonGreedyDiscreteExplorerNode, EpsilonGreedyContinuousExplorerNode,
                       GaussianContinuousExplorereNode, BoltzmannDiscreteExplorerNode)

def test_explorer_nodes():

    prob_vec = [0.1,0.1,0.6,0.2]
    inp = mdp.numx.random.randn(10000,1)
    node = DiscreteExplorerNode(n_actions=len(prob_vec), prob_vec=prob_vec)
    out = node.execute(inp)
    obs_prob_vec = mdp.numx.histogram(out, bins=range(len(prob_vec)+1))[0]/float(inp.shape[0])
    assert(out.shape == inp.shape)
    assert_array_almost_equal(prob_vec, obs_prob_vec, decimal=2)

    inp = mdp.numx.random.randn(10000, 2)
    node = ContinuousExplorerNode(action_lims=[(-1, 1), (0, 2)])
    out = node.execute(inp)
    assert (out.shape == inp.shape)

    node = ContinuousExplorerNode(action_lims=[(-1, 1), (0, 2)])
    out = node.execute(inp)
    assert (out.shape == inp.shape)


    inp = mdp.numx.random.randint(0,6,(10000,1)).astype('float')
    node = EpsilonGreedyDiscreteExplorerNode(n_actions=6, epsilon=0., decay=1.0)
    out = node.execute(inp)
    assert_array_equal(inp, out)

    node = EpsilonGreedyDiscreteExplorerNode(n_actions=6, epsilon=1., decay=0.0)
    out = node.execute(inp)
    assert_array_equal(inp, out)

    inp = mdp.numx.ones((10000, 2)) * mdp.numx.array([[0., 1.]])
    node = EpsilonGreedyContinuousExplorerNode(action_lims=[(-1, 1), (0, 2)], epsilon=0., decay=1.0)
    out = node.execute(inp)
    assert_array_equal(inp, out)

    node = EpsilonGreedyContinuousExplorerNode(action_lims=[(-1, 1), (0, 2)], epsilon=1., decay=0.0)
    out = node.execute(inp)
    assert_array_equal(inp, out)

    inp = mdp.numx.ones((10000, 3)) * mdp.numx.array([[0., 1., 0.]])
    node = BoltzmannDiscreteExplorerNode(n_actions=3, temperature=0., decay=1.0)
    out = node.execute(inp)
    assert_array_equal(mdp.numx.ones((inp.shape[0], 1)), out)

    inp = mdp.numx.ones((10000, 2)) * mdp.numx.array([[0., 1.]])
    node = GaussianContinuousExplorereNode(action_lims=[(-1, 1), (0, 2)], sigma=0, decay=1.0)
    out = node.execute(inp)
    assert_array_equal(inp, out)
