
import mdp

class DiscreteExplorerNode(mdp.OnlineNode):
    """
    DiscreteExplorerNode is an online explorer node for reinforcement
    learning that returns a discrete random action upon each execute call.

    The train method does nothing for this class. Subclasses can overwrite
    "_train" method to add any additionaly functionality.

    The execute function returns a discrete action. Optionally, one can also
    provide a probability vector to specify a probability distribution when
    sampling actions at random.

    """
    def __init__(self, n_actions, prob_vec=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        n_actions - Number of actions.
        prob_vec - Probability of individual action selection (default is uniform).
        """
        super(DiscreteExplorerNode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.n_actions=n_actions
        if prob_vec is None:
            self._prob_vec = mdp.numx.ones(self.n_actions) / float(self.n_actions)
        else:
            self._prob_vec = prob_vec

        self._output_dim = 1

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

    def _roulette_wheel(self, prob_vec):
        rnd_nr = self.numx_rng.rand()
        pi = 0.
        pj = 0.
        indx = None
        for i in xrange(len(prob_vec)):
            pj = prob_vec[i]
            if (pi <= rnd_nr) and (rnd_nr < (pi + pj)):
                indx = i
                break
            else:
                pi += pj
        return indx

    def _train(self, x):
        pass

    def _execute(self, x, prob_vec=None):
        if prob_vec is None:
            prob_vec = self._prob_vec
        if isinstance(prob_vec, list) or (prob_vec.ndim == 1):
            return mdp.numx.reshape([self._roulette_wheel(prob_vec) for _ in xrange(x.shape[0])], [x.shape[0],1])
        else:
            if prob_vec.shape != (x.shape[0],self.n_actions):
                raise mdp.NodeException("prob_vec has wrong shape, given %s required %s"%(str(prob_vec.shape), str((x.shape[0], self.n_actions))))
            return mdp.numx.reshape([self._roulette_wheel(prob_vec[i]) for i in xrange(x.shape[0])], [x.shape[0], 1])


class ContinuousExplorerNode(mdp.OnlineNode):
    """
    ContinuousExplorerNode is an online explorer node for reinforcement
    learning that returns a continuous random action upon each execute call.

    The train method does nothing for this class. Subclasses can overwrite
    "_train" method to add any additionaly functionality.

    The execute function returns a discrete action. Optionally, one can also
    provide a probability distribution function for sampling actions at random.

    """
    def __init__(self, action_lims, momentum=0., dstr_fn=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        action_lims - Lower and upper bounds for all action dimensions. Eg. [(min1, min2, ...,), (max1, max2, ...,)]
        momentum - Momemtum parameter (mass) for selecting actions.
        dstr_fn - A funtion that returns a real number given the min bound and max bound.
                    Eg. dstr_fn(action_lims[0], action_lims[1], shape) -> array, where mini <= array_elem_i < maxi.
        """
        super(ContinuousExplorerNode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        #check action_lims
        if len(action_lims) != 2:
            raise mdp.NodeException("'action_lims' must contain two elements of a list or a tuple of lower bounds and upper bounds respectively and not %d. "%(len(action_lims)),
                                    " Eg. [(min1, min2, ...,), (max1, max2, ...,)].")
        if len(action_lims[0]) != len(action_lims[1]):
            raise mdp.NodeException("Length of lower_bounds ('action_lims[0]=%d') does not match the length of upper_bounds ('action_lims[1]=%d)."%(len(action_lims[0]), len(action_lims[1])))

        self.action_lims = action_lims
        self._m = momentum
        self._dstr_fn = dstr_fn
        self._action = None

        self._output_dim = len(action_lims[0])

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

    @property
    def dstr_fn(self):
        return self._dstr_fn

    @dstr_fn.setter
    def dstr_fn(self, fn):
        if fn is None:
            return
        else:
            if not callable(fn):
                raise mdp.NodeException("Given dstr_fn is not callable. It must of a function that returns an action array.")
            # check if the samples returned are within the specified bounds.
            a = fn(self.action_lims[0], self.action_lims[1], (10,self.input_dim))
            for i in xrange(self.input_dim):
                if (a[:,i].min() < self.action_lims[0][i]) or (a[:,i].min() >= self.action_lims[1][i]):
                    raise mdp.NodeException("Out of bounds error. Given dstr_fn returns an action %d outside a specified"
                                            "bounds (%d, %d) " % (a[:,i].min(), self.action_lims[0][i], self.action_lims[1][i]))
            # if no exceptions raised
            self._dstr_fn = fn

    def _check_params(self, x):
        if self._dstr_fn is None:
            self.dstr_fn = self.numx_rng.uniform

        if self._action is None:
            # prev action
            self._action = self.dstr_fn(self.action_lims[0], self.action_lims[1], [1, self.input_dim])

    def _train(self, x):
        pass

    def _execute(self, x):
        if self._m == 0:
            return self._dstr_fn(self.action_lims[0], self.action_lims[1], x.shape)
        else:
            action = mdp.numx.zeros((x.shape[0]+1, self.output_dim))
            action[0] = self._action
            for i in xrange(x.shape[0]):
                new_action = self.dstr_fn(self.action_lims[0], self.action_lims[1], [1, self.input_dim])
                action[i+1] = self._m*action[i] + (1-self._m)*new_action
            self._action = action[-1]
            return action[1:]


class EpsilonGreedyDiscreteExplorerNode(DiscreteExplorerNode):
    """
    EpsilonGreedyDiscreteExplorerNode is an online explorer node for reinforcement
    learning that switches between exploration (random action) and exploitation (given action)
    modes based on the current value of a decaying epsilon parameter (0<epsilon<1).

    The node returns a random action with a probability of epsilon and the given action
    with the probability of (1-epsilon)

    The train function of this node only updates the epsilon parameter. The execute function
    returns a discrete action. Optionally, one can also provide a probability vector to
    specify a probability distribution when sampling actions at random.

    For more information on epsilon greedy approach refer
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning:
    An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.

    """
    def __init__(self, n_actions, epsilon=1., decay=0.999, prob_vec=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        epsilon - Parameter that balances exploration vs exploitation.
        decay - Decay constant of epsilon. Epsilon decays exponentially.
        """
        super(EpsilonGreedyDiscreteExplorerNode, self).__init__(n_actions=n_actions, prob_vec=prob_vec, input_dim=input_dim,
                                                                output_dim=output_dim, dtype=dtype, numx_rng=numx_rng)
        self.epsilon=epsilon
        self.decay=decay


    def _train(self, x):
        self.epsilon*=self.decay**x.shape[0]

    def _execute(self, x, prob_vec=None):
        f = (self.numx_rng.rand(x.shape[0],1) < self.epsilon)
        return f*super(EpsilonGreedyDiscreteExplorerNode, self)._execute(x, prob_vec) + (1-f)*x


class EpsilonGreedyContinuousExplorerNode(ContinuousExplorerNode):
    """
    EpsilonGreedyContinuousExplorerNode is an online explorer node for reinforcement
    learning that switches between exploration (random action) and exploitation (given action)
    modes based on the current value of a decaying epsilon parameter (0<epsilon<1).

    The node returns a random action with a probability of epsilon and the given action
    with the probability of (1-epsilon)

    The train function of this node only updates the epsilon parameter. The execute function
    returns a continuous action. Optionally, one can also provide a probability distribution
    function for sampling actions at random.

    For more information on epsilon greedy approach refer
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning:
    An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.

    """
    def __init__(self, action_lims, epsilon=1., decay=0.999, momentum=0., dstr_fn=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        epsilon - Parameter that balances exploration vs exploitation.
        decay - Decay constant of epsilon. Epsilon decays exponentially.
        """
        super(EpsilonGreedyContinuousExplorerNode, self).__init__(action_lims, momentum=momentum, dstr_fn=dstr_fn,
                                                                  input_dim=input_dim, output_dim=output_dim, dtype=dtype,numx_rng=numx_rng)

        self.epsilon=epsilon
        self.decay=decay


    def _train(self, x):
        self.epsilon*=self.decay**x.shape[0]

    def _execute(self, x):
        f = (self.numx_rng.rand(x.shape[0],1) < self.epsilon)
        return f*super(EpsilonGreedyContinuousExplorerNode, self)._execute(x) + (1-f)*x



class BoltzmannDiscreteExplorerNode(DiscreteExplorerNode):
    """
    BoltzmannDiscreteExplorerNode is an online explorer node for reinforcement
    learning that balances exploration (random action) and exploitation (optimal action)
    modes based on a decaying temperature parameter.

    The train function of this node only updates the temperature parameter.
    The input to the node is a action/state value-vector for each action, the output is a softmax-function
    output weighted by the temperature parameter.

    For more information on epsilon greedy approach refer
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning:
    An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.

    """


    def __init__(self, n_actions, temperature=50., decay=0.999, input_dim=None, output_dim=None, dtype=None,numx_rng=None):
        """
        temperature - Parameter that balances exploration vs exploitation.
        decay - Decay constant of epsilon. Epsilon decays exponentially.
        """
        super(BoltzmannDiscreteExplorerNode, self).__init__(n_actions=n_actions, prob_vec=None, input_dim=input_dim,
                                                                output_dim=output_dim, dtype=dtype, numx_rng=numx_rng)
        self.temperature = temperature
        self.decay = decay

        # input 'x' must be a vector of action or state values for each action.
        self._input_dim = n_actions
        self._output_dim = 1

    def _train(self, x):
        self.temperature*=self.decay**x.shape[0]

    def _execute(self, x, prob_vec=None):
        e = mdp.numx.e
        if self.temperature < 0.01:
            return mdp.numx.argmax(x, axis=1)[:,None]
        else:
            _p = x/self.temperature
            prob_vec = mdp.numx.power(e, _p)/mdp.numx.sum(mdp.numx.power(e,_p), axis=1)[:,None]
            return super(BoltzmannDiscreteExplorerNode, self)._execute(x, prob_vec=prob_vec)


class GaussianContinuousExplorereNode(ContinuousExplorerNode):
    """
    GaussianContinuousExplorerNode is an online explorer node for reinforcement
    learning that balances exploration (random action) and exploitation (given action)
    modes based on a decaying standard deviation parameter (sigma).

    The train function of this node only updates the sigma parameter.
    The input to the node is a action/state value-vector for each action, the output is a softmax-function
    output weighted by the temperature parameter.

    For more information on epsilon greedy approach refer
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning:
    An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.

    """
    def __init__(self, action_lims, sigma=None, decay=0.999, momentum=0., dstr_fn=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        epsilon - Parameter that balances exploration vs exploitation.
        decay - Decay constant of epsilon. Epsilon decays exponentially.
        """
        super(GaussianContinuousExplorereNode, self).__init__(action_lims, momentum=momentum, dstr_fn=dstr_fn,
                                                                  input_dim=input_dim, output_dim=output_dim, dtype=dtype,numx_rng=numx_rng)

        if sigma is None:
            self.sigma=mdp.numx.ones(len(action_lims))
        elif mdp.numx.isscalar(sigma):
            self.sigma=mdp.numx.ones(len(action_lims))*sigma
        self.decay=decay

        self._input_dim = len(action_lims)
        self._output_dim = len(action_lims)

    def _train(self, x):
        self.sigma*=self.decay**x.shape[0]

    def _execute(self, x):
        return x + self.numx_rng.multivariate_normal(mdp.numx.zeros(self.input_dim), mdp.numx.diag(self.sigma), x.shape[0])

