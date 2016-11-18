
from _tools import *
from mdp.nodes import PG2DNode, PGCurveNode, PGImageNode
import os


def test_pg_nodes():
    if os.environ.has_key('DISPLAY')  and (os.environ['DISPLAY'] != ''):
        for dispnode in [PG2DNode(), PGCurveNode(), PGImageNode(img_xy=(10,10))]:
            x = mdp.numx_rand.randn(1 ,100)
            y = dispnode(x)
            assert dispnode._viewer.is_alive()
            dispnode.close()
            assert not dispnode._viewer.is_alive()
            assert_array_equal(x,y)

