from __future__ import print_function

import argparse
import os
import numpy as np
import skvideo.io

import torch
import torch.multiprocessing as mp
from torch.optim import Adam

from visdom import Visdom

from envs import create_vizdoom_env
from model import ActorCritic
from test import test
from train import train
from optim import SharedAdam

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('run')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001 * 2.5,
    help='learning rate (default: 0.00025)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--entropy-coef',
    type=float,
    default=0.0005,
    help='entropy term coefficient (default: 0.0005)')
parser.add_argument(
    '--value-loss-coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--conv-depth-loss-coef',
    type=float,
    default=10,
    help='conv depth loss coefficient (default: 10)')
parser.add_argument(
    '--lstm-depth-loss-coef',
    type=float,
    default=10,
    help='lstm depth loss coefficient (default: 10)')
parser.add_argument(
    '--max-grad-norm',
    type=float,
    default=50,
    help='value loss coefficient (default: 50)')
parser.add_argument(
    '--seed', type=int, default=666, help='random seed (default: 666)')
parser.add_argument(
    '--num-processes',
    type=int,
    default=4,
    help='how many training processes to use (default: 4)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=50,
    help='number of forward steps in A3C (default: 50)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=20,
    help='logging interval (default: 20)')
parser.add_argument(
    '--max-episode-steps',
    type=int,
    default=10**8,
    help='maximum steps of a training run (default: 100000000)')
parser.add_argument(
    '--config-path',
    default='./doomfiles/default.cfg',
    help='ViZDoom configuration path (default: ./doomfiles/default.cfg)')
parser.add_argument(
    '--train-scenario-path',
    default='./doomfiles/11.wad',
    help='ViZDoom scenario path for training (default: ./doomfiles/11.wad)')
parser.add_argument(
    '--test-scenario-path',
    default='./doomfiles/11.wad',
    help='ViZDoom scenario path for testing (default: ./doomfiles/11.wad)')
parser.add_argument(
    '--no-shared',
    default=False,
    help='use an optimizer without shared momentum.')
parser.add_argument(
    '--save-interval',
    type=int,
    default=20,
    help='save model every n episodes (default: 20)')
parser.add_argument(
    '--eval-interval',
    type=int,
    default=60,
    help='run evaluation every n seconds (default: 60)')
parser.add_argument(
    '--topology', action='store_true', help='Use topological mapping.')
parser.add_argument('--checkpoint-path', help='file path to save models')
parser.add_argument('--video-path', help='file path to save video')
parser.add_argument(
    '--visdom-port', type=int, default=8097, help='visdom port')
args = parser.parse_args()


def build_logger(build_state, checkpoint={}, run='NavA3C', port=8097):
    vis = Visdom(port=port)
    env = run
    wins = mp.Manager().dict()
    offset = checkpoint.setdefault('offset', -1) + 1

    if 'plots' in checkpoint:
        wins = dict((name, id) for name, id in checkpoint['plots'].items()
                    if vis.win_exists(id, env))

    def _save_checkpoint(step):
        if step % args.save_interval != 0 or args.checkpoint_path is None:
            return

        checkpoint_path = os.path.abspath(args.checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        state = build_state()
        state['plots'] = dict(wins)
        state['offset'] = offset
        torch.save(state, args.checkpoint_path)

    def _log_scatter(value, step, win_name, title, mode='train'):
        if mode == 'test':
            step += offset
        elif step % args.log_interval != 0:
            return

        if not vis.check_connection():
            return

        if isinstance(value, torch.Tensor):
            norm = value.numpy()
        else:
            norm = value

        win_id = wins.setdefault(win_name)
        if win_id is None:
            wins[win_name] = vis.scatter(
                X=np.array([[step, norm]]),
                win=win_id,
                env=env,
                opts=dict(title=title))
        else:
            vis.scatter(
                X=np.array([[step, norm]]),
                win=win_id,
                env=env,
                update='append')
        vis.save([env])

    def _log_reward(total_reward, step, mode):
        if mode == 'train' and step % args.log_interval != 0:
            return

        if not vis.check_connection():
            return

        win_name = 'total_reward_{}'.format(mode)
        win_id = wins.setdefault(win_name)
        if win_id is None:
            wins[win_name] = vis.line(
                np.array([0, 0]),
                win=win_id,
                env=env,
                opts=dict(title='{} reward'.format(mode)))
        else:
            vis.line(
                Y=np.array([total_reward]),
                X=np.array([step]),
                win=win_id,
                env=env,
                update='append')

        vis.save([env])

    def _log_video(video, step):
        step += offset

        if args.video_path is None:
            return

        video_path = os.path.abspath(args.video_path)
        video_dir = os.path.dirname(video_path)

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        skvideo.io.vwrite(video_path, np.array(video))

        if not vis.check_connection():
            return

        win_name = 'last_test_episode'
        win_id = wins.setdefault(win_name)
        if win_id is None:
            wins[win_name] = vis.video(
                videofile=video_path,
                win=win_id,
                env=env,
                opts=dict(title='episode {}'.format(step)))
        else:
            vis.video(
                videofile=video_path,
                win=win_id,
                env=env,
                opts=dict(title='episode {}'.format(step)))

        vis.save([env])

    return dict(
        video=_log_video,
        grad_norm=lambda n, s: _log_scatter(n, s, 'grad_norm', 'gradient norm'),
        train_reward=lambda r, s: _log_reward(r, s, 'train'),
        test_reward=lambda r, s: _log_reward(r, s, 'test'),
        train_time=
        lambda n, s: _log_scatter(n, s, 'train_time', 'training wall time (per episode)', 'train'),
        test_time=
        lambda n, s: _log_scatter(n, s, 'test_time', 'evaluation wall time (per episode)', 'test'),
        checkpoint=_save_checkpoint)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    kill = mp.Event()
    counter = mp.Value('i', 0)
    steps = mp.Value('i', 0)
    lock = mp.Lock()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = create_vizdoom_env(args.config_path, args.train_scenario_path)
    shared_model = ActorCritic(env.observation_space.spaces[0].shape[0],
                               env.action_space, args.topology)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = Adam(shared_model.parameters(), lr=args.lr)
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        counter.value = checkpoint['episodes']
        shared_model.load_state_dict(checkpoint['model'])
        shared_model.share_memory()
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.share_memory()
    else:
        checkpoint = {}

    processes = []

    logging = build_logger(lambda: dict(episodes=counter.value,
                                        model=shared_model.state_dict(),
                                        optimizer=optimizer.state_dict()),
                           checkpoint,
                           args.run,
                           args.visdom_port)

    p = mp.Process(
        target=test,
        args=(args.num_processes, args, shared_model, (counter, steps),
              logging, kill))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(
            target=train,
            args=(rank, args, shared_model, (counter, steps), lock, optimizer,
                  logging, kill))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
