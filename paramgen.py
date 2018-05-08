from __future__ import print_function

import argparse
import os
import numpy as np
from textwrap import dedent

parser = argparse.ArgumentParser()
parser.add_argument('root_path')
parser.add_argument('config_name')
parser.add_argument('email')


def main(args):
    file_path = os.path.dirname(os.path.realpath(__file__))
    root_base = os.path.abspath(args.root_path)

    if not os.path.isdir(root_base):
        print('root path is not a directory')

    checkpoint_dir = os.path.join(root_base, 'checkpoint')
    video_dir = os.path.join(root_base, 'media')
    visdom_dir = os.path.join(root_base, 'visdom')

    if os.path.isfile(checkpoint_dir) or os.path.isfile(video_dir):
        print('remove {} / {}'.format(checkpoint_dir, video_dir))

    hyperparams = dict(lr=np.random.uniform(np.power(10., -4), 5 * np.power(10., -4)),
                       entropy_coef=np.random.uniform(np.power(10., -4), np.power(10., -3)),
                       config_path='{}/doomfiles/default.cfg'.format(file_path),
                       train_scenario_path='{}/doomfiles/11.wad'.format(file_path),
                       test_scenario_path='{}/doomfiles/11.wad'.format(file_path),
                       num_steps=50,
                       conv_depth_loss_coef=np.random.choice([1 / 3.0, 10, 33]),
                       lstm_depth_loss_coef=np.random.choice([1, 10 / 3.0, 10]),
                       save_interval=1000,
                       log_interval=10,
                       num_processes=16,
                       checkpoint_path=os.path.join(root_base, 'checkpoint', args.config_name) + '.ckpt',
                       video_path=os.path.join(root_base, 'media', args.config_name) + '.mp4')

    headers = dedent("""\
    #PBS -N {}
    #PBS -j oe
    #PBS -l walltime=60:00:00
    #PBS -l nodes=1:ppn=20
    #PBS -S /bin/bash
    #PBS -m abe
    #PBS -M {}
    #PBS -V
    """.format(args.config_name, args.email))

    visdom = dedent("""\
    mkdir -p {path}
    
    if lsof -Pi :6666 -sTCP:LISTEN -t >/dev/null ; then
        echo "visdom running"
    else
        echo "visdom not running, launching"
        python -m visdom.server -env_path={path} -port=6666 &
    fi
    """.format(path=visdom_dir))

    for l in (headers + visdom).splitlines():
        r = l.rstrip()

        if r:
            print(r)
    print("python {}/main.py {} \\".format(file_path, args.config_name))
    for idx, (flag, value) in enumerate(hyperparams.items()):
        print(" " * 10 + "{:<50} {}".format('--{}={}'.format(flag.replace("_", "-"), value),
                                            '\\' if idx + 1 < len(hyperparams) else str()))


if __name__ == "__main__":
    main(parser.parse_args())
