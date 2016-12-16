
import mdp
from mdp.utils import mult


class CaclaRLNode(mdp.RLNode):
    """
    CaclaRLNode implements the Continuous Actor-Critic Learning Automaton reinforcement learning algorithm.

    """

    def __init__(self, observation_dim, action_dim, alpha=0.1, beta=0.1, gamma=0.99, output_mode='action',
                 input_dim=None, output_dim=None, dtype=None, numx_rng=None):
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
                                          input_dim=input_dim, output_dim=output_dim, dtype=dtype,
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

        self._cache = {'td_err': 0., 'value_params': None, 'policy_params': None}

    def _check_params(self, x):
        if self._theta is None:
            self._theta = 0.1 * self.numx_rng.randn(self.observation_dim, 1)

        if self._psi is None:
            self._psi = 0.1 * self.numx_rng.randn(self.observation_dim, self.action_dim)

    def get_value(self, phi):
        """Returns state value(s)."""
        return mult(phi, self._theta)

    def get_action(self, phi):
        """Returns greedy action(s)."""
        return mult(phi, self._psi)

    def _train(self, *args):
        phi, phi_, a, r, done = args[:5]
        td_err = r + self._gamma * self.get_value(phi_) - self.get_value(phi)
        grad_theta = self._alpha * mult(phi.T, td_err)
        grad_psi = self._beta * mult(phi.T, (td_err > 0) * (a - self.get_action(phi)))
        self._theta += grad_theta
        self._psi += grad_psi

        self.cache['td_err'] = td_err
        self.cache['value_params'] = self._theta.copy()
        self.cache['policy_params'] = self._psi.copy()

    def _execute(self, *args):
        phi_ = args[1]
        if self.output_mode == 'action':
            return self.get_action(phi_)
        elif self.output_mode == 'value':
            return self.get_value(phi_)
        else:
            raise mdp.NodeException("Unrecognized 'output_mode'. Accepted values: 'action' and "
                                    "'value', given %s." % str(self.output_mode))
