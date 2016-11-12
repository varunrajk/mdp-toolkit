
from mdp.online.nodes import SignalAvgNode
from mdp.test._tools import *

def test_signalavgnode():
    node = SignalAvgNode()
    mu = numx.random.uniform(-3,3,5)
    x = numx.random.multivariate_normal(mean=mu,cov=numx.identity(len(mu)),size=1000)
    node.train(x)
    assert_almost_equal(x.mean(axis=0), node.get_average().ravel())

    node = SignalAvgNode(avg_n=300)
    node.train(x)
    assert_array_almost_equal_diff(x.mean(axis=0), node.get_average().ravel(), digits=1)
