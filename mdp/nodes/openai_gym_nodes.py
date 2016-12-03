
import mdp
from difflib import SequenceMatcher
import gym

class GymNode(mdp.OnlineNode):
    """GymNode is a thin OnlineNode wrapper over OpenAi's Gym library.
    For more information about the Gym library refer to the
    documentation provided at https://gym.openai.com/

    This node is a non-trainable node. The node's execute call takes
    an action array as an input and returns a flattened array of
    (current state (observation), future state, action, reward, done flag).

    The node's metaclass is selected to be an OnlineNode instead of a Node
    in order to enforce a shared numx_rng between the node and the gym's
    environment.

    The node also provides additional utility methods:
    'get_random_samples' - generates a required number of output
    samples for random selected actions, starting from the current state.
    The env is reset back to the current state.
    'stop_rendering' - closes gym's rendering window if initialized.

    **Instance variables of interest**

      ``self.state_dim / self.action_dim``
         Flattened state (observation) / action dimension

      ``self.state_shape / self.action_shape``
         The original state shape / action shape. Eg. (100,80,3) for an RGB image.

      ``self.state_type / self.action_type``
         Discrete or continuous state / action space.

      ``self.state_lims / self.action_lims``
         Upper and lower bounds of state / action space.

      ``n_states / n_actions``
         Number of states or actions for discrete types.

    """
    def __init__(self, env_name, render=False, auto_reset=True, numx_rng=None):
        """
        env_name - Registered gym environment name. Eg. "MountainCar-v0"
        render - Enable or disable rendering. Disabled by default.
        auto_reset - Automatically resets the environment if gym env's done is True.
        """
        super(GymNode, self).__init__(input_dim=None, output_dim=None, dtype=None, numx_rng=None)

        self._env_registry = gym.envs.registry.env_specs
        if self._env_registry.has_key(env_name):
            self._env_name = env_name
        else:
            similar_envs_str = '(' + ', '.join(self._get_similar_env_names(env_name)) + ')'
            raise mdp.NodeException(
                "Unregistered environment name. Are you looking for any of these?: \n%s" % (similar_envs_str))
        self.env = gym.make(self.env_name)

        self.render = render
        self.auto_reset = auto_reset

        # set a shared numx_rng
        self.numx_rng = numx_rng

        # get state dims and shape
        if isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
            self.state_type = 'discrete'
            self.state_dim = 1
            self.state_shape = (1,)
            self.n_states = self.env.observation_space.n
            self.state_lims = [[0],[self.n_states-1]]
        else:
            self.state_type = 'continuous'
            self.state_shape = self.env.observation_space.shape
            self.state_dim = mdp.numx.product(self.state_shape)
            self.n_states = None
            self.state_lims = [self.env.observation_space.low, self.env.observation_space.high]

        # get action dims
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_type = 'discrete'
            self.action_dim = 1
            self.action_shape = (1,)
            self.n_actions = self.env.action_space.n
            self.action_lims = [[0],[self.n_actions-1]]
        else:
            self.action_type = 'continuous'
            self.action_shape = self.env.action_space.shape
            self.action_dim = mdp.numx.product(self.action_shape)
            self.n_actions = None
            self.action_lims = [self.env.action_space.low, self.env.action_space.high]

        # set input_dim
        self._input_dim = self.action_dim

        # set output dims
        self._output_dim = self.state_dim*2 + self.action_dim + 1 + 1

        # get observation
        self._s = self.env.reset().reshape(1, self.state_dim)

        # cache to store variables
        self._cache = {'info': None}

    # properties

    @property
    def env_name(self):
        return self._env_name   #read only

    def _get_similar_env_names(self, name):
        keys = self._env_registry.keys()
        ratios = [SequenceMatcher(None, name, key).ratio() for key in keys]
        return [x for (y, x) in sorted(zip(ratios, keys), reverse=True)][:5]

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

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
                a = mdp.numx.asscalar(a)
            s, r, done, info = self.env.step(a)
            if self.render:
                self.env.render()
            if self.auto_reset and done:
                self.env.reset()
            yield s, r, done, info

    def _execute(self, x):
        s_, r, done, info = zip(*self.__steps(x))
        s_ = mdp.numx.reshape(s_, [len(s_), self.state_dim])
        s = mdp.numx.vstack((self._s, s_[:-1]))
        self._s = s_[-1:]
        r = mdp.numx.reshape(r, [len(r),1])
        a = x
        done = mdp.numx.reshape(done, [len(done),1])
        y = mdp.numx.hstack((s,s_,a,r,done))
        self.cache['info'] = info[-1]
        return y

    # utility methods

    def stop_rendering(self):
        # stop gym's rendering if active
        self.env.render(close=True)

    def get_random_samples(self, n=1):
        # Generates random samples without changing the
        # current state of the environment.
        x = mdp.numx.asarray([self.env.action_space.sample() for _ in xrange(n)]).reshape(n, self.input_dim)
        s_, r, done, info = zip(*self.__steps(x))
        s_ = mdp.numx.reshape(s_, [len(s_), self.state_dim])
        s = mdp.numx.vstack((self._s, s_[:-1]))
        r = mdp.numx.reshape(r, [len(r),1])
        a = x
        done = mdp.numx.reshape(done, [len(done),1])
        y = mdp.numx.hstack((s,s_,a,r,done))
        return y

