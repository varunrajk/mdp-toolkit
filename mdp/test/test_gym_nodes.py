
from mdp.nodes import GymNode
from ._tools import *

requires_gym = skip_on_condition("not mdp.config.has_gym","This test requires OpenAi's Gym Library")

@requires_gym
def test_gym_nodes():
    gym_node = GymNode('MountainCar-v0', render=False, numx_rng=mdp.numx_rand.RandomState(seed=13))
    a = mdp.numx.array([[0], [0], [1]]).astype('float')
    out = gym_node(a)
    assert_array_equal(out[:,gym_node.observation_dim*2:gym_node.observation_dim*2+gym_node.action_dim], a)
    assert (out.shape[1] == gym_node.observation_dim*2+gym_node.action_dim+1+1)
    assert(gym_node.numx_rng == gym_node.env.np_random)

