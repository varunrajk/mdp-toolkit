
from mdp.nodes import MovingTimeDiffNode
from ._tools import *

def test_movingtimediffnode():
    node = MovingTimeDiffNode()
    x = mdp.numx_rand.randn(10,10)
    out=[]
    for i in xrange(x.shape[0]):
        node.train(x[i:i+1])
        out.append(node.execute(x[i:i+1]))
    assert_array_equal(mdp.numx.asarray(out).squeeze()[1:], x[1:]-x[:-1])

