import cv2
import gym
import torch
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
        game.init()
        self.game = game
        self.scenario = scenario
        self.wad = WAD(scenario)

        num_buttons = len(game.get_available_buttons())
        self.action_space = gym.spaces.Discrete(num_buttons)
        self.action_map = tuple([action_idx == button_idx for button_idx in range(num_buttons)]
                                for action_idx in range(num_buttons))
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(0, 1, (3, 82, 82), dtype=np.float32),
                                                   gym.spaces.Box(0, 1, (8, 4 * 16), dtype=np.float32),
                                                   gym.spaces.Box(-1, 1, (0,), dtype=np.float32),
                                                   gym.spaces.Discrete(num_buttons),
                                                   gym.spaces.Box(-1, 1, (3,), dtype=np.float32)))
        self.episode_reward = 0.0
        self.step_counter = 0
        self.seed()
        self.reset()

    def _state(self):
        state = self.game.get_state()

        # Camera Input
        if state:
            screen_buffer = np.moveaxis(state.screen_buffer, 0, -1)
            screen_buffer = screen_buffer[:, 20:20 + 120]
            screen_buffer = cv2.resize(screen_buffer, (82, 82))
            screen_buffer = np.moveaxis(screen_buffer, -1, 0)
            screen_buffer = screen_buffer.astype(np.float32)
            screen_buffer /= 255.

            # Depth
            bins = [0, 0.05, 0.175, 0.3, 0.425, 0.55, 0.675, 0.8, 1]
            depth_buffer = state.depth_buffer
            depth_buffer = depth_buffer[60:80, :]
            depth_buffer = cv2.resize(depth_buffer, (4, 16))
            depth_buffer = depth_buffer.astype(np.float32)
            depth_buffer /= 255.
            depth_buffer = np.power(1. - depth_buffer, 10)
            depth_buffer = np.digitize(depth_buffer, bins)
            depth_buffer = np.eye(len(bins))[depth_buffer]
        else:
            screen_buffer = np.zeros((3, 84, 84))
            depth_buffer = np.zeros((1, 4, 16))

        # Reward
        last_reward = np.array([self.game.get_last_reward()], dtype=np.float32)

        # Action
        last_action = np.array(self.game.get_last_action(), dtype=np.float32)

        # Velocity
        velocity = np.array([self.game.get_game_variable(gamevar)
                             for gamevar in (vizdoom.GameVariable.VELOCITY_X,
                                             vizdoom.GameVariable.VELOCITY_Y,
                                             vizdoom.GameVariable.VELOCITY_Z)],
                            dtype=np.float32)

        return screen_buffer, depth_buffer, last_reward, last_action, velocity

    def seed(self, seed=None):
        if seed is not None:
            self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        reward = self.game.make_action(self.action_map[np.asscalar(action)])
        done = self.game.is_episode_finished()
        state = self._state()
        self.episode_reward += reward
        self.step_counter += 1
        return state, reward, done, {}

    def reset(self):
        next_map = np.random.choice(self.wad.maps.keys())
        self.game.new_episode(next_map)
        self.episode_reward = 0.0
        self.step_counter = 0
        return self._state()

    def render(self, mode='rgb_array'):
        if mode == 'human':
            plt.figure(1)
            plt.clf()
            plt.imshow(self.render(mode='rgb_array'))
            plt.pause(0.001)
            return None

        if mode == 'rgb_array':
            return self._state()

        if mode == 'rgbd_array':
            return self._state()

        assert False, 'Unsupported render mode'


def create_vizdoom_env(config, scenario):
    env = ViZDoomEnv(config, scenario)
    return env


def state_to_torch(state):
    return (torch.from_numpy(t).unsqueeze(0) for t in state)
