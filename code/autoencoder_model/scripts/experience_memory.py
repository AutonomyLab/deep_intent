""" Implements data structures to save and sample previous
observations, actions, rewards and terminals.

RingBuffer implementation is from
https://github.com/matthiasplappert/keras-rl/
"""

import numpy as np

class RingBuffer(object):
  def __init__(self, max_length):
    self.max_length = max_length
    self.start = 0
    self.length = 0
    self.data = [None] * self.max_length

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    if index < 0 or index >= self.length:
      raise KeyError('Index: {}'.format(index))
    return self.data[(self.start + index) % self.max_length]

  def append(self, value):
    if self.length < self.max_length:
      self.length += 1
    elif self.length == self.max_length:
      self.start = (self.start + 1) % self.max_length

    self.data[(self.start + self.length - 1) % self.max_length] = value

class ExperienceMemory(object):

    def __init__(self, memory_length=100):
        self.memory_length = memory_length
        self.observations = RingBuffer(memory_length)

    def save_experience(self, observation):
        self.observations.append(observation)

    def get_exp_window(self, window_vid_nums):
        observations = []
        # The terminality of the last observation does not affect
        # computation. The last observation of the first window (second
        # index in this loop) determines the action, reward and
        # and terminality of the window.
        for i in window_vid_nums:
            # Terminals at an earlier index than end - 1 belong to another
            # episode.
            observations.append(self.observations[i])

        return np.asarray(observations)

    def sample_minibatch(self, window_size):

        last_index = len(self.observations) - 1
        # window_vid_nums = np.random.randint(0, last_index, size=window_size)
        # Sample without replacement; No repetitions
        window_vid_nums = np.random.choice(last_index, window_size, replace=False)
        # always include the latest observation for training
        # window_vid_nums[-1] = last_index
        observations = self.get_exp_window(window_vid_nums)

        return observations

    def get_trainable_fakes(self, current_gens, exp_window_size):
        n_current_gens = current_gens.shape[0]
        for i in range(n_current_gens):
            self.save_experience(current_gens[i])

        exp_samples = self.sample_minibatch(window_size=exp_window_size)

        return np.concatenate((current_gens[0: n_current_gens-exp_window_size], exp_samples), axis=0)


