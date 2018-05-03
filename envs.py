import os
import cv2
import gym
import numpy as np
import vizdoom
from gym.spaces.box import Box
import matplotlib.pyplot as plt
from omg import WAD


class ViZDoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'rgbd_array']}

    def __init__(self, config, scenario):
        game = vizdoom.DoomGame()
        game.load_config(config)
        game.set_doom_scenario_path(scenario)
        game.set_mode(vizdoom.Mode.PLAYER)
        game.init()
        self.game = game
        self.scenario = scenario
        self.wad = WAD(scenario)

        num_buttons = len(game.get_available_buttons())
        self.action_space = gym.spaces.Discrete(num_buttons)
        self.actions = [[action_idx == button_idx for button_idx in range(num_buttons)]
                        for action_idx in range(num_buttons)]
        self.observation_space = gym.spaces.Box(0, 255, [game.get_screen_channels(),
                                                         game.get_screen_height(),
                                                         game.get_screen_width()], dtype=np.float32)
        self.episode_reward = 0.0
        self.step_counter = 0
        self.seed()
        self.reset()

    def _get_state(self):
        return self.game.get_state().screen_buffer

    def seed(self, seed=None):
        if seed is not None:
            self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = self._get_state() if not done else None
        self.episode_reward += reward
        self.step_counter += 1
        return state, reward, done, {}

    def reset(self):
        self.game.new_episode(np.random.choice(self.wad.maps.keys()))
        self.episode_reward = 0.0
        self.step_counter = 0
        return self.game.get_state()

    def render(self, mode='rgb_array'):
        if mode == 'human':
            plt.figure(1)
            plt.clf()
            plt.imshow(self.render(mode='rgb_array'))
            plt.pause(0.001)
            return None

        if mode == 'rgb_array':
            return self._get_state()

        if mode == 'rgbd_array':
            return self._get_state()

        assert False, 'Unsupported render mode'


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


def ceeate_vizdoom_env(config, scenario):
    env = ViZDoomEnv(config, scenario)
    env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
                          observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
                         observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
