import os
import random
import shutil
import argparse


def sample_and_copy_dirs(src_dir, dst_dir, k=1000, seed=0):
    random.seed(seed)

    os.makedirs(dst_dir, exist_ok=True)

    # 找到所有子目录
    dirs = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) or d.endswith('.json')
    ]

    if len(dirs) < k:
        raise ValueError(f"Only {len(dirs)} directories found, but {k} requested")

    # 随机采样
    sampled = random.sample(dirs, k)

    # 复制目录
    for d in sampled:
        src_path = os.path.join(src_dir, d)
        dst_path = os.path.join(dst_dir, d)

        shutil.copy(src_path, dst_path)
        print(f"Copied: {d}")

    print(f"\nDone. {k} directories copied to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source folder")
    parser.add_argument("--dst", required=True, help="destination folder")
    parser.add_argument("--num", type=int, default=1000, help="number of directories")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    sample_and_copy_dirs(args.src, args.dst, args.num, args.seed)
