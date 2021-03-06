import mdp
from mdp.utils import mult


class CaclaRLNode(mdp.RLNode):
    """
    CaclaRLNode implements the Continuous Actor-Critic Learning Automaton reinforcement learning algorithm.
    More information about CACLA can be found in "Reinforcement learning in continuous action spaces",
    H. V. Hasselt and M. A. Wiering, Approximate Dynamic Programming and Reinforcement Learning, pp 272--279,
    ADPRL 2007.

    **Instance variables of interest**

      ``self.td_err``
         temporal difference error

    """

    def __init__(self, observation_dim, action_dim, alpha=0.1, beta=0.1, gamma=0.99, output_mode='action',
                 input_dim=None, dtype=None, numx_rng=None):
        """
        Additional arguments:

        alpha - critic's learning rate

        beta - actor's learning rate

        gamma - reward discount factor

        output_mode - Determines the return value of the execute method. Accepted values:
                    'action' - returns greedy action(s) for the given observation(s)
                    'value' - returns state-value(s) for the given observation(s)
        """

        super(CaclaRLNode, self).__init__(observation_dim=observation_dim, action_dim=action_dim, reward_dim=1,
                                          input_dim=input_dim, output_dim=None, dtype=dtype,
                                          numx_rng=numx_rng)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._theta = None
        self._psi = None

        if output_mode not in ['action', 'value']:
            raise mdp.NodeException("Unrecognized 'output_mode'. Accepted values: 'action' and "
                                    "'value', given %s." % str(output_mode))

        self.output_mode = output_mode
        # set output_dim
        if self.output_mode == 'action':
            self.output_dim = self.action_dim
        elif self.output_mode == 'value':
            self.output_dim = 1

        self.td_err = 0.0

    def _check_params(self, x):
        if self._theta is None:
            self._theta = 0.1 * self.numx_rng.randn(self.observation_dim, 1).astype(self.dtype)

        if self._psi is None:
            self._psi = 0.1 * self.numx_rng.randn(self.observation_dim, self.action_dim).astype(self.dtype)

    def get_value(self, phi):
        """Returns state value(s)."""
        return mult(phi, self._theta)

    def get_action(self, phi):
        """Returns greedy action(s)."""
        return mult(phi, self._psi)

    def _train(self, x):
        phi, phi_, a, r, done = self._split_x(x)
        td_err = r + self._gamma * self.get_value(phi_) - self.get_value(phi)
        grad_theta = self._alpha * mult(phi.T, td_err)
        grad_psi = self._beta * mult(phi.T, (td_err > 0) * (a - self.get_action(phi)))
        self._theta += grad_theta
        self._psi += grad_psi
        self.td_err = td_err

    def get_value_params(self):
        """Returns value function parameters"""
        return self._theta.copy()

    def get_policy_params(self):
        """Returns policy function parameters"""
        return self._psi.copy()

    def _execute(self, x):
        phi_ = self._split_x(x)[1]
        if self.output_mode == 'action':
            return self.get_action(phi_)
        elif self.output_mode == 'value':
            return self.get_value(phi_)
        else:
            raise mdp.NodeException("Unrecognized 'output_mode'. Accepted values: 'action' and "
                                    "'value', given %s." % str(self.output_mode))

    def __repr__(self):
        name = type(self).__name__
        observation_dim = "observation_dim=%s" % str(self.observation_dim)
        action_dim = "action_dim=%s" % str(self.action_dim)
        alpha = "alpha=%s" % str(self._alpha)
        beta = "beta=%s" % str(self._beta)
        gamma = "gamma=%s" % str(self._gamma)
        output_mdoe = "output_mode='%s'" % str(self.output_mode)
        input_dim = "input_dim=%s" % str(self.input_dim)
        dtype = "dtype=%s" % str(self.dtype)
        numx_rng = "numx_rng=%s" % str(self.numx_rng)
        args = ', '.join((observation_dim, action_dim, alpha, beta, gamma, output_mdoe, input_dim, dtype, numx_rng))
        return name + '(' + args + ')'
