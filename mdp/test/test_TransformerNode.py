
from _tools import *
from mdp.nodes import TransformerNode


def test_transformer_node():
    from scipy.misc import face

    inp = face()[None, :]
    inp = mdp.numx.vstack((inp, inp))
    inp_shape = inp.shape[1:]
    inp = inp.reshape(2, 768*1024*3).astype('float')

    # check with no transformations
    tr_node = TransformerNode(input_shape=inp_shape)
    out = tr_node(inp)
    assert_array_equal(inp, out)

    # check with a sequence of transformation
    tr_node = TransformerNode(input_shape=inp_shape, transform_seq=['gray', 'img_255_1'])
    out = tr_node(inp)
    assert(out.max() <= 1)
    assert(out.min() >= 0)
    assert(out.shape == (inp.shape[0], inp.shape[1]/3))

    #TODO test sklearn preprrocessing