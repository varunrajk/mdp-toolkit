
from ._tools import *
from mdp import RLNode, OnlineNode
from mdp.nodes import BasisFunctionNode, QRLNode, QLambdaRLNode


class BogusEnv(OnlineNode):
    def __init__(self, n_observations, input_dim=None, output_dim=None,
                 dtype=None, numx_rng=None):
        super(BogusEnv, self).__init__(input_dim, output_dim, dtype, numx_rng)

        self.n_observations = n_observations
        self.observation_lims = [[0], [n_observations-1]]
        self._actions = [-1, 1]
        self._r = 5
        self.state = None

    @staticmethod
    def is_trainable():
        return False

    def _check_params(self, x):
        if self.state is None:
            self.state = self.numx_rng.randint(0, self.n_observations)

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

    def _execute(self, action):
        action = action.astype('int')
        obs = []
        for a in action:
            a = a[0]
            fut_state = self.state + self._actions[a]
            fut_state = mdp.numx.clip(fut_state, 0, self.n_observations)
            r = 0.0
            done = 0
            if fut_state == self._r:
                r = 1.0
                done = 1
            obs += [[self.state, fut_state, a, r, done]]
            self.state = fut_state
        return mdp.numx.asarray(obs)


class BogusRLNode(RLNode):
    def _train(self, *args):
        self.phi, self.phi_, self.a, self.r, self.done = args[:5]

    def _execute(self, *args):
        return args[:5]


def test_rlnode():
    rng = mdp.numx_rand.RandomState(seed=0)
    env = BogusEnv(n_observations=10)

    # Features
    sbfn = mdp.nodes.BasisFunctionNode('indicator', env.observation_lims)
    cln = mdp.hinet.CloneLayer(node=sbfn, n_nodes=2)
    features = mdp.hinet.Layer(nodes=[cln, mdp.nodes.IdentityNode(input_dim=3)])

    node = BogusRLNode(observation_dim=sbfn.output_dim, action_dim=1, numx_rng=rng)

    assert(node.numx_rng == rng)
    a = rng.randint(0, 2, [20, 1])
    node.train(features(env(a)))

    assert (node.phi.shape == (1, sbfn.output_dim))
    assert (node.phi_.shape == (1, sbfn.output_dim))
    assert (node.a.shape == (1, 1))
    assert (node.r.shape == (1, 1))
    assert (node.done.shape == (1, 1))

    phi, phi_, a, r, done = node(features(env(a)))

    assert (phi.shape == (a.shape[0], sbfn.output_dim))
    assert (phi_.shape == (a.shape[0], sbfn.output_dim))
    assert (a.shape == (a.shape[0], 1))
    assert (r.shape == (a.shape[0], 1))
    assert (done.shape == (a.shape[0], 1))

    node = BogusRLNode(observation_dim=sbfn.output_dim, action_dim=1, numx_rng=rng)
    node.set_training_type('batch')
    a = rng.randint(0, 2, [20, 1])
    node.train(features(env(a)))

    assert (node.phi.shape == (a.shape[0], sbfn.output_dim))
    assert (node.phi_.shape == (a.shape[0], sbfn.output_dim))
    assert (node.a.shape == (a.shape[0], 1))
    assert (node.r.shape == (a.shape[0], 1))
    assert (node.done.shape == (a.shape[0], 1))


def test_qrlnode():
    rng = mdp.numx_rand.RandomState(seed=0)

    env = BogusEnv(n_observations=10)

    # Features
    sbfn = BasisFunctionNode('indicator', env.observation_lims)
    cln = mdp.hinet.CloneLayer(node=sbfn, n_nodes=2)
    features = mdp.hinet.Layer(nodes=[cln, mdp.nodes.IdentityNode(input_dim=3)])

    gamma = 0.99
    q = QRLNode(observation_dim=sbfn.output_dim, n_actions=2, alpha=1., gamma=gamma,
                output_mode='value', numx_rng=rng)

    phis = mdp.numx.eye(10)
    a = rng.randint(0, 2, [10000, 1])
    q.train(features(env(a)))
    qvals = q.get_value(phis)

    assert (tuple(qvals.argmax(axis=0)) == (6, 4))
    for i in xrange(1, 8):
        if i < 6:
            assert mdp.numx.isclose(qvals[i, 0], gamma * qvals[i + 1, 0], atol=1e-10)
        else:
            assert mdp.numx.isclose(gamma * qvals[i, 0], qvals[i + 1, 0], atol=1e-10)
        if i < 4:
            assert mdp.numx.isclose(qvals[i, 1], gamma * qvals[i + 1, 1], atol=1e-10)
        else:
            assert mdp.numx.isclose(gamma * qvals[i, 1], qvals[i + 1, 1], atol=1e-10)


def test_qlambdarlnode():
    rng = mdp.numx_rand.RandomState(seed=0)

    env = BogusEnv(n_observations=10)

    # Features
    sbfn = BasisFunctionNode('indicator', env.observation_lims)
    cln = mdp.hinet.CloneLayer(node=sbfn, n_nodes=2)
    features = mdp.hinet.Layer(nodes=[cln, mdp.nodes.IdentityNode(input_dim=3)])

    q = QLambdaRLNode(observation_dim=sbfn.output_dim, n_actions=2, alpha=0.3, gamma=0.99,
                      traces_lambda=0.8, output_mode='value', numx_rng=rng)

    phis = mdp.numx.eye(10)
    a = rng.randint(0, 2, [10000, 1])
    q.train(features(env(a)))
    qvals = q.get_value(phis)

    assert (tuple(qvals.argmax(axis=0)) == (6, 4))
    pi_opt = mdp.numx.ones([10, 1])
    pi_opt[6:] = 0
    assert_array_equal(q.get_action(phis), pi_opt)
