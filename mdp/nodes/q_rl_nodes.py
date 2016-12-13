
import mdp
from mdp.utils import mult


class QRLNode(mdp.RLNode):
    """
    QRLNode implements the Q reinforcement learning algorithm.

    Supports only discrete actions.

    """
    def __init__(self, observation_dim, n_actions, alpha=1., gamma=0.99, output_mode='action',
                 dtype=None, numx_rng=None):
        """
        Additional arguments:

        n_actions - number of actions.

        alpha - learning rate

        gamma - reward discount factor

        output_mode - Determines the return value of the execute method. Accepted values:
                    'action' - returns greedy action(s) for the given observation(s)
                    'value' - returns q-values for all actions for the given observation(s)
        """
        super(QRLNode, self).__init__(observation_dim=observation_dim, action_dim=1, reward_dim=1,
                                            dtype=dtype, numx_rng=numx_rng)
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._theta = None
        if output_mode not in ['action', 'value']:
            raise mdp.NodeException("Unrecognized 'output_mode'. Accepted values: 'action' and "
                                    "'value', given %s." % str(output_mode))
        self.output_mode = output_mode
        # set output_dim
        if self.output_mode == 'action':
            self.output_dim = 1
        elif self.output_mode == 'value':
            self.output_dim = self._n_actions

        self.output_mode = output_mode
        self._cache = {'td_err': 0., 'value_params': None}

    @property
    def n_actions(self):
        return self._n_actions

    def _check_params(self, x):
        if self._theta is None:
            self._theta = 0.*self.numx_rng.randn(self.observation_dim, self.n_actions)

    def get_value(self, phi, a=None):
        """Returns q value(s)."""
        if a is not None:
            if mdp.numx.isscalar(a):
                return mult(phi, self._theta[:, a])
            else:
                return (phi * self._theta[:, a.ravel()].T).sum(axis=1, keepdims=True)
        else:
            return mult(phi, self._theta)

    def get_action(self, phi):
        """Returns greedy action(s)."""
        return self.get_value(phi).argmax(axis=1)[:, None]

    def _train(self, *args):
        phi, phi_, a, r, done = args[:5]
        a = a.astype('int')

        q = self.get_value(phi, a)
        q_ = self.get_value(phi_).max(axis=1, keepdims=True)

        td_err = r + self._gamma * q_ - q

        # assign td err for the actions taken
        td_err_a = mdp.numx.zeros([len(a), self.n_actions])
        td_err_a[range(len(a)), a.ravel()] = td_err.ravel()

        grad_theta = self._alpha * mult(phi.T, td_err_a)
        self._theta += grad_theta

        self.cache['td_err'] = td_err
        self.cache['value_params'] = self._theta.copy()

    def _execute(self, *args):
        phi_ = args[1]
        if self.output_mode == 'action':
            return self.get_action(phi_)
        elif self.output_mode == 'value':
            return self.get_value(phi_)
            # qvals = mdp.numx.asarray([self.get_qvalue(phi_[i]) for i in xrange(len(phi_))])
            # return qvals
        else:
            raise mdp.NodeException("Unrecognized 'output_mode'. Accepted values: 'action' and "
                                    "'value', given %s." % str(self.output_mode))


class QLambdaRLNode(QRLNode):
    """
    QLambdaRLNode implements the Q reinforcement learning algorithm with eligibility traces (Peng's Qlambda).

    Supports only discrete actions.

    """
    def __init__(self, observation_dim, n_actions, alpha=1., gamma=0.99, traces_lambda=0., output_mode='action',
                 dtype=None, numx_rng=None):
        """
        Additional arguments:

        traces_lambda - eligibility trace parameter (lambda)
        """
        super(QLambdaRLNode, self).__init__(observation_dim=observation_dim, n_actions=n_actions, alpha=alpha,
                                            gamma=gamma, output_mode=output_mode, dtype=dtype, numx_rng=numx_rng)
        self._lambda = traces_lambda
        self._e = None

    def _check_params(self, x):
        super(QLambdaRLNode, self)._check_params(x)

        if self._e is None:
            self._e = mdp.numx.zeros([self.observation_dim, self.n_actions])

    def _train(self, *args):
        phi, phi_, a, r, done = args[:5]
        a = a.astype('int')

        q = self.get_value(phi, a)
        q_ = self.get_value(phi_).max(axis=1, keepdims=True)

        td_err = r + self._gamma * q_ - q

        grad_theta = 0.0
        for i in xrange(len(a)):
            self._e[:, a[i]] += phi[i:i + 1, :].T
            grad_theta += self._alpha * td_err[i] * self._e
            self._e *= self._gamma * self._lambda
            if done[i]:
                # reset traces
                self._e = mdp.numx.zeros([self.observation_dim, self.n_actions])

        self._theta += grad_theta

        self.cache['td_err'] = td_err
        self.cache['value_params'] = self._theta.copy()
