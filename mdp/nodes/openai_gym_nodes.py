import mdp
from difflib import SequenceMatcher
import gym
from gym import spaces


class GymNode(mdp.OnlineNode):
    """GymNode is a thin OnlineNode wrapper over OpenAi's Gym library.
    For more information about the Gym library refer to the
    documentation provided at https://gym.openai.com/

    This node is a non-trainable node. The node's execute call takes
    an action array as an input and returns a flattened array of
    (current observation (observation), future observation, action, reward, done flag).

    The node's metaclass is selected to be an OnlineNode instead of a Node
    in order to enforce a shared numx_rng between the node and the gym's
    environment.

    The node also provides additional utility methods:
    'stop_rendering' - closes gym's rendering window if initialized.

    'get_environment_samples' - generates a required number of output
    samples for randomly selected actions, starting from the current env state.
    Note that the env state changes after the method is called.
    This is equivalent to an execute call without any input.

    'get_random_actions' - returns a random set of valid actions
    within the environment.

    **Instance variables of interest**

      ``self.observation_dim / self.action_dim``
         Flattened observation  / action dimension

      ``self.observation_shape / self.action_shape``
         The original observation shape / action shape. Eg. (100,80,3) for an RGB image.

      ``self.observation_type / self.action_type``
         Discrete or continuous observation / action space.

      ``self.observation_lims / self.action_lims``
         Upper and lower bounds of observation / action space.

      ``n_observations / n_actions``
         Number of observations or actions for discrete types.

    """

    def __init__(self, env_name, render=False, auto_reset=True, dtype=None, numx_rng=None):
        """
        env_name - Registered gym environment name. Eg. "MountainCar-v0"
        render - Enable or disable rendering. Disabled by default.
        auto_reset - Automatically resets the environment if gym env's done is True.
        """
        super(GymNode, self).__init__(input_dim=None, output_dim=None, dtype=dtype, numx_rng=None)

        self._env_registry = gym.envs.registry.env_specs
        if env_name in self._env_registry.keys():
            self._env_name = env_name
        else:
            similar_envs_str = '(' + ', '.join(self._get_similar_env_names(env_name)) + ')'
            raise mdp.NodeException(
                "Unregistered environment name. Are you looking for any of these?: \n%s" % similar_envs_str)
        self.env = gym.make(self.env_name)

        self.render = render
        self.auto_reset = auto_reset

        # set a shared numx_rng
        self.numx_rng = numx_rng

        # get observation dims and shape
        if isinstance(self.env.observation_space, spaces.discrete.Discrete):
            self.observation_type = 'discrete'
            self.observation_dim = 1
            self.observation_shape = (1,)
            self.n_observations = self.env.observation_space.n
            self.observation_lims = [[0], [self.n_observations - 1]]
        else:
            self.observation_type = 'continuous'
            self.observation_shape = self.env.observation_space.shape
            self.observation_dim = mdp.numx.product(self.observation_shape)
            self.n_observations = None
            self.observation_lims = [self.env.observation_space.low, self.env.observation_space.high]

        # get action dims
        if isinstance(self.env.action_space, spaces.discrete.Discrete):
            self.action_type = 'discrete'
            self.action_dim = 1
            self.action_shape = (1,)
            self.n_actions = self.env.action_space.n
            self.action_lims = [[0], [self.n_actions - 1]]
        else:
            self.action_type = 'continuous'
            self.action_shape = self.env.action_space.shape
            self.action_dim = mdp.numx.product(self.action_shape)
            self.n_actions = None
            self.action_lims = [self.env.action_space.low, self.env.action_space.high]

        # set input_dim
        self._input_dim = self.action_dim

        # set output dims
        self._output_dim = self.observation_dim * 2 + self.action_dim + 1 + 1

        # get observation
        self._phi = mdp.numx.reshape(self.env.reset(), [1, self.observation_dim])

        # cache to store variables
        self._cache = {'info': None}

    # properties

    @property
    def env_name(self):
        return self._env_name  # read only

    def _get_similar_env_names(self, name):
        keys = self._env_registry.keys()
        ratios = [SequenceMatcher(None, name, key).ratio() for key in keys]
        return [x for (y, x) in sorted(zip(ratios, keys), reverse=True)][:5]

    def _set_numx_rng(self, rng):
        """Set a shared numx random number generator.
        """
        self.env.np_random = rng
        self._numx_rng = rng

    # Node capabilities

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    # environment steps
    def __steps(self, x):
        for a in x:
            if self.action_type == 'discrete':
                a = int(mdp.numx.asscalar(a))
            phi, r, done, info = self.env.step(a)
            if self.render:
                self.env.render()
            if self.auto_reset and done:
                self.env.reset()
            yield phi, r, done, info

    def _execute(self, x):
        phi_, r, done, info = zip(*self.__steps(x))
        phi_ = mdp.numx.reshape(phi_, [len(phi_), self.observation_dim])
        phi = mdp.numx.vstack((self._phi, phi_[:-1]))
        self._phi = phi_[-1:]
        r = mdp.numx.reshape(r, [len(r), 1])
        a = x
        done = mdp.numx.reshape(done, [len(done), 1])
        y = mdp.numx.hstack((phi, phi_, a, r, done))
        self.cache['info'] = info[-1]
        return self._refcast(y)

    def _train(self, x):
        pass

    # utility methods

    def stop_rendering(self):
        # stop gym's rendering if active
        self.env.render(close=True)

    def get_random_actions(self, n=1):
        return self._refcast(mdp.numx.reshape([self.env.action_space.sample() for _ in xrange(n)], (n, self.input_dim)))

    def get_environment_samples(self, n=1):
        # Generates random environment samples
        return self.execute(self.get_random_actions(n))
