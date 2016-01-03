# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def forward_probability(time_t, state, observation, *model_parameters):
    """

    :param model_parameters:
    :param time_t:
    :param observation:
    :param state:
    :return:
    """
    pi, a, b = model_parameters
    obs = observation
    if state > len(pi):
        raise ValueError('state value out of range!')

    if time_t == 1:
        return pi[state-1] * b[state-1, obs[0]]

    states = range(len(pi))
    alpha1 = [pi[i]*b[i, obs[0]] for i in states]
    alpha2 = alpha1.copy()
    for t in range(1, time_t-1):
        for i in states:
            alpha2[i] = sum([alpha1[j]*a[j, i] for j in states]) * b[i, obs[t]]
        alpha1 = alpha2.copy()

    return sum([alpha1[j] * a[j, state-1] for j in states]) * b[state-1, obs[time_t-1]]


def backward_probability(time_t, state, observation, *model_parameters):
    """

    :param time_t:
    :param state:
    :param observation:
    :param model_parameters:
    :return:
    """
    pi, a, b = model_parameters
    obs = observation

    if state > len(pi):
        raise ValueError('state value is out of range!')

    if time_t == len(obs):
        return 1.0

    states = range(len(pi))
    beta1 = np.ones(len(pi))
    beta2 = beta1.copy()

    for t in range(len(obs)-1, time_t, -1):
        for i in states:
            beta2[i] = sum([a[i, j] * b[j, obs[t]] * beta1[j] for j in states])
        beta1 = beta2.copy()

    return sum([a[state-1, j] * b[j, obs[time_t-1]] * beta1[j] for j in states])


def expectation_prob(observation, *model_parameters):
    pi, a, b = model_parameters
    obs = observation
    states = range(len(pi))
    time = len(obs)

    if len(obs) == 1:
        return sum([pi[i]*b[i, obs[0]] for i in states])

    alpha1 = [pi[i]*b[i, obs[0]] for i in states]
    alpha2 = alpha1.copy()
    for t in range(1, time):
        for i in states:
            alpha2[i] = sum([alpha1[j]*a[j, i] for j in states]) * b[i, obs[t]]
        alpha1 = alpha2.copy()

    return sum(alpha2)



if __name__ == '__main__':
    #         state 1,    2,    3,    4
    pi = np.array([0.2, 0.4, 0.4])
    a = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])
    b = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    obs = [0, 1, 0]
    print(backward_probability(1, 3, obs, *(pi, a, b)))

