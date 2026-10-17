"""Train the SceneFoundry eps baseline on ARScene data inside a Slurm GPU job."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training_configs/scenefoundry_livingroom_unconditional_eps_900k.yaml')
    parser.add_argument('--output')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--max-steps', type=int, help='Total optimizer updates, including resumed steps')
    parser.add_argument('--resume', nargs='?', const='auto', default='auto', metavar='PATH|auto|none')
    args = parser.parse_args()
    from networks.scenefoundry.step_training import train
    train(yaml.safe_load(Path(args.config).read_text()), output=args.output,
          device_name=f'cuda:{args.cuda}', max_steps=args.max_steps, resume=args.resume)


if __name__ == '__main__':
    main()
