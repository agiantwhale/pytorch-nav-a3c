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
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-torch-threads', type=int,
                    help='Number of torch threads')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--log-interval', type=int, default=20,
                    help='logging interval (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--config-path', default='./doomfiles/default.cfg',
                    help='ViZDoom configuration path (default: ./doomfiles/default.cfg)')
parser.add_argument('--train-scenario-path', default='./doomfiles/3.wad',
                    help='ViZDoom scenario path for training (default: ./doomfiles/3.wad)')
parser.add_argument('--test-scenario-path', default='./doomfiles/3.wad',
                    help='ViZDoom scenario path for testing (default: ./doomfiles/3.wad)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--save-interval', type=int, default=20,
                    help='save model every n episodes (default: 20)')
parser.add_argument('--checkpoint-path', help='file path to save models')
parser.add_argument('--video-path', help='file path to save video')
args = parser.parse_args()


def build_logger(build_state, checkpoint=None):
    vis = Visdom()
    env = 'NavA3C'
    wins = dict()

    if checkpoint and 'plots' in checkpoint:
        wins = checkpoint['plots']

    def _save_checkpoint(step):
        if step % args.save_interval != 0 or args.checkpoint_path is None:
            return
        state = build_state()
        state['plots'] = wins
        torch.save(state, args.checkpoint_path)

    def _log_grad_norm(grad_norm, step):
        if step % args.log_interval != 0 or not vis.check_connection():
            return
        norm = grad_norm.numpy()
        win_id = wins.setdefault('grad_norm')
        if win_id is None or vis.win_exists(win_id):
            wins['grad_norm'] = vis.scatter(X=np.array([[step, norm]]), win=win_id, env=env,
                                            opts=dict(title='gradient norm'))
        else:
            vis.scatter(X=np.array([[step, norm]]), win=win_id, env=env, update='append')
        vis.save([env])

    def _log_reward(total_reward, step, mode):
        if step % args.log_interval != 0:
            return
        if not vis.check_connection():
            return
        win_name = 'total_reward_{}'.format(mode)
        win_id = wins.setdefault(win_name)
        if win_id is None or vis.win_exists(win_id):
            wins[win_name] = vis.line(np.array([0, 0]), win=win_id, env=env,
                                      opts=dict(title='{} reward'.format(mode)))
        else:
            vis.line(Y=np.array([total_reward]), X=np.array([step]),
                     win=win_id, env=env, update='append')

        vis.save([env])

    def _log_video(video, step):
        if step % args.log_interval != 0 or not vis.check_connection():
            return
        if args.video_path is None:
            return

        videofile = args.video_path
        if os.path.isfile(videofile):
            os.remove(videofile)

        skvideo.io.vwrite(videofile, np.array(video))

    return dict(video=_log_video,
                grad_norm=_log_grad_norm,
                train_reward=lambda r, s: _log_reward(r, s, 'train'),
                test_reward=lambda r, s: _log_reward(r, s, 'test'),
                checkpoint=_save_checkpoint)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    if args.num_torch_threads:
        torch.set_num_threads(args.num_torch_threads)

    torch.manual_seed(args.seed)
    env = create_vizdoom_env(args.config_path, args.train_scenario_path)
    shared_model = ActorCritic(env.observation_space.spaces[0].shape[0], env.action_space)
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
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        checkpoint = None

    processes = []

    logging = build_logger(lambda: dict(episodes=counter.value,
                                        model=shared_model.state_dict(),
                                        optimizer=optimizer.state_dict()),
                           checkpoint)

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, logging))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer, logging))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
