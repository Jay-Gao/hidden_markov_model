# !/usr/local/bin python3
# -*- coding: utf-8 -*-

import numpy as np
import collections

from numpy.linalg import norm
from numpy import random


def sampling(prob_distribution):
    """
    generate a sample for given probability distribution.
    :param prob_distribution: discrete probability distribution.
    """
    sum_distribution = np.array(prob_distribution).cumsum()
    t = random.random()
    for state in range(len(sum_distribution)):
        if sum_distribution[state] > t:
            return state


class HiddenMarkovModel(object):

    def __init__(self):
        self._pi = None
        self._a = None
        self._b = None
        self._obs = []
        self._states = []

    def set_model_parameters(self, *model_parameters):
        pi, a, b = model_parameters
        self._pi = np.array(pi)
        self._a = np.array(a)
        self._b = np.array(b)

    def get_obs(self, observations=None, states=None):
        if observations is None:
            pass
        else:
            try:
                for obs in observations:
                    if isinstance(obs, collections.Iterable):
                        self._obs.append(obs)
                    else:
                        self._obs.append(observations)
                        break
            except AttributeError:
                print('observations must be Iterable!')
        if states is not None:
            try:
                for state in states:
                    if isinstance(state, collections.Iterable):
                        self._states.append(state)
                    else:
                        self._states.append(states)
                        break
            except AttributeError:
                print('states must be None or Iterable')

    def forward_probability(self, time_t, state, obs):
        """
        """
        if state > len(self._pi):
            raise ValueError('state value out of range!')

        if time_t == 1:
            return self._pi[state-1] * self._b[state-1, obs[0]-1]

        states = range(len(self._pi))
        alpha1 = [self._pi[i]*self._b[i, obs[0]-1] for i in states]
        alpha2 = alpha1.copy()
        for t in range(1, time_t-1):
            for i in states:
                alpha2[i] = sum([alpha1[j]*self._a[j, i] for j in states]) * self._b[i, obs[t]-1]
            alpha1 = alpha2.copy()

        return sum([alpha1[j] * self._a[j, state-1] for j in states]) * self._b[state-1, obs[time_t-1]-1]

    def backward_probability(self, time_t, state, obs):
        """

        :param time_t:
        :param state:
        :return:
        """
        if state > len(self._pi) or state < 1:
            raise ValueError('state value is out of range!')

        if time_t == len(obs):
            return 1.0

        states = range(len(self._pi))
        beta1 = np.ones(len(self._pi))
        beta2 = beta1.copy()

        for t in range(len(obs)-1, time_t, -1):
            for i in states:
                beta2[i] = sum([self._a[i, j] * self._b[j, obs[t]-1] * beta1[j] for j in states])
            beta1 = beta2.copy()

        return sum([self._a[state-1, j] * self._b[j, obs[time_t]-1] * beta1[j] for j in states])

    def expectation_prob(self, obs):

        states = range(len(self._pi))
        time = len(obs)

        if len(obs) == 1:
            return sum([self._pi[i]*self._b[i, obs[0]-1] for i in states])

        alpha1 = [self._pi[i]*self._b[i, obs[0]-1] for i in states]
        alpha2 = alpha1.copy()
        for t in range(1, time):
            for i in states:
                alpha2[i] = sum([alpha1[j]*self._a[j, i] for j in states]) * b[i, obs[t]-1]
            alpha1 = alpha2.copy()

        return sum(alpha2)

    def gama(self, time_t, state, obs):
        """
        the probability of state at time t for given observation and model parameters.
        :param time_t: time t
        :param state: state at time t
        :return:
        """
        alpha = self.forward_probability(time_t, state, obs)
        beta = self.backward_probability(time_t, state, obs)
        Exp = self.expectation_prob(obs)

        return alpha * beta / Exp

    def zita(self, t, state_i, state_j, obs):
        """
        the probability state is i at t and state is j at t+1 for given observation and mode parameters.
        :param t:
        :param state_i:
        :param state_j
        :param observation:
        :param model_parameters:
        :return:
        """
        i = state_i
        j = state_j
        alpha = self.forward_probability(t, i, obs)
        beta = self.backward_probability(t+1, j, obs)
        Exp = self.expectation_prob(obs)

        return alpha * self._a[i-1, j-1] * self._b[j-1, obs[t]-1] * beta / Exp

    def expectation_of_state(self, state, obs):
        """
        calculate the expectation of given state under the circumstance of observation.
        """
        Exp = self.expectation_prob(obs)
        return sum((self.forward_probability(t, state, obs) * self.backward_probability(t, state, obs)\
                    for t in range(1, len(obs)+1))) / Exp

    def expectation_of_transform_from_state(self, state, obs):
        """
        calculate the expectation of transforming from given state under the circumstance of observation.
        """
        Exp = self.expectation_prob(obs)
        return sum((self.forward_probability(t, state, obs) * self.backward_probability(t, state, obs)\
                    for t in range(1, len(obs)))) / Exp

    def expectation_of_transform_from_state_to_state(self, i, j, obs):
        """
        calculate the expectation of transforming from i to j.
        """
        Exp = self.expectation_prob(obs)
        return sum((self.zita(t, i, j, obs) for t in range(1, len(obs)))) / Exp

    def generate(self, num_of_obs=100, num_of_test=10, latent_state=False):
        """
        generate observations for given model.
        """
        states = []
        obs = []

        for test in range(num_of_test):
            tmp_states = []
            tmp_obs = []
            i = sampling(self._pi)
            j = sampling(self._b[i])
            tmp_states.append(i+1)
            tmp_obs.append(j+1)

            for k in range(num_of_obs-1):
                i = sampling(self._a[i])
                j = sampling(self._b[i])
                tmp_states.append(i+1)
                tmp_obs.append(j+1)

            states.append(tmp_states)
            obs.append(tmp_obs)

        # default: ignore latent variable states.
        if latent_state is False:
            return obs
        else:
            return obs, states

    def fit(self, observations=None, states=None, num_states=10, num_obs_states=5):

        self.get_obs(observations, states)

        if len(self._states) > 0:
            # states are not empty, a supervised model.
            PI = np.zeros(num_states)
            self._pi = PI.copy()
            A = np.zeros((num_states, num_states))
            self._a = A.copy()
            B = np.zeros((num_states, num_obs_states))
            self._b = B.copy()

            for state in self._states:
                PI[state[0]-1] += 1
                for t in range(len(state)-1):
                    i = state[t] - 1
                    j = state[t+1] - 1
                    A[i, j] += 1
            for state, obs in zip(self._states, self._obs):
                for i, j in zip(state, obs):
                    B[i-1, j-1] += 1

            for i in range(num_states):
                self._pi[i] = float(PI[i]) / sum(PI)
                denominator_a = sum(A[i])
                denominator_b = sum(B[i])
                for j in range(num_states):
                    self._a[i, j] = float(A[i, j]) / denominator_a
                for j in range(num_obs_states):
                    self._b[i, j] = float(B[i, j]) / denominator_b

            return self._pi, self._a, self._b

        else:
            # states are empty, a supervised model. Using Baum-Welch method.
            pi = [0.3, 0.3, 0.4]
            a = np.array([
                [0.3, 0.3, 0.4],
                [0.3, 0.3, 0.4],
                [0.3, 0.3, 0.4],
            ])
            b = np.array([
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ])

            eps = 1e-8
            # initialize model parameters.
            self.set_model_parameters(pi, a, b)
            error = 10.0
            while error > eps:
                # denominator_pi = sum(self.expectation_prob(obs) for obs in self._obs)
                denominator_pi = 0.0
                for i in range(num_states):
                    denominator_pi += sum(self.forward_probability(1, i+1, obs)*self.backward_probability(1, i+1, obs)\
                                          for obs in self._obs)
                for i in range(num_states):
                    pi[i] = sum(self.forward_probability(1, i+1, obs)*self.backward_probability(1, i+1, obs)\
                                for obs in self._obs) / denominator_pi

                    denominator_a = 0.0
                    for obs in self._obs:
                        for t in range(1, len(obs)):
                            denominator_a += self.forward_probability(t, i+1, obs)*\
                                             self.backward_probability(t, i+1, obs)
                    for j in range(num_states):
                        a_ij = 0.0
                        for obs in self._obs:
                            for t in range(1, len(obs)):
                                a_ij += self.forward_probability(t, i+1, obs)*self._a[i, j]*self._b[j, obs[t]-1]*\
                                        self.backward_probability(t+1, j+1, obs)
                        a[i, j] = a_ij / denominator_a

                    denominator_b = 0.0
                    for obs in self._obs:
                        for t in range(1, len(obs)+1):
                            denominator_b += self.forward_probability(t, i+1, obs)*\
                                             self.backward_probability(t, i+1, obs)
                    for k in range(num_obs_states):
                        b_ik = 0.0
                        for obs in self._obs:
                            for t in range(1, len(obs)+1):
                                b_ik += self.forward_probability(t, i+1, obs)*self.backward_probability(t, i+1, obs)\
                                        if obs[t-1]-1 == k else 0.0
                        b[i, k] = b_ik / denominator_b

                error = norm(self._pi-pi)**2 + norm(self._a-a)**2 + norm(self._b-b)**2
                self.set_model_parameters(pi, a, b)

            return self._pi, self._a, self._b

#########################################################################################################


if __name__ == '__main__':
    #         state 1,   2,   3
    pi = np.array([0.2, 0.4, 0.4])
    a = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])
    b = np.array([
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
    ])

    hmm = HiddenMarkovModel()
    hmm.set_model_parameters(pi, a, b)
    obs, states = hmm.generate(num_of_obs=20, num_of_test=5, latent_state=True)

    pi1, a1, b1 = hmm.fit(obs, num_states=3, num_obs_states=2)
    print(pi1)
    print(a1)
    print(b1)


