import os
import json
import argparse
import numpy as np
from collections import Counter
from glob import glob
from utils import THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
import tqdm

def extract_classes_from_json(data):
    """
    专门适配你的格式：
    data["rooms"][i]["objList"][j]["coarseSemantic"]
    """
    classes = []

    if "rooms" not in data:
        return classes

    for room in data["rooms"]:
        if "objList" not in room:
            continue

        for obj in room["objList"]:
            if "coarseSemantic" in obj and not obj['coarseSemantic'] == 'Door' and not obj['coarseSemantic'] == 'Window':
                if obj["coarseSemantic"] not in THREED_FRONT_CATEGORY:
                    c = THREED_FRONT_FURNITURE[obj["coarseSemantic"]]
                    classes.append(c)
                else:
                    classes.append(obj["coarseSemantic"])


    return classes


def build_distribution(json_dir):
    counter = Counter()

    json_files = glob(os.path.join(json_dir, "*.json"))

    for path in tqdm.tqdm(json_files):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            classes = extract_classes_from_json(data)
            counter.update(classes)

        except Exception as e:
            print(f"[Warning] {path} failed: {e}")

    return counter


def counter_to_prob(counter, all_keys, eps=1e-8):
    vec = np.array([counter.get(k, 0) for k in all_keys], dtype=np.float64)

    vec = vec + eps  # 防止0
    vec = vec / vec.sum()

    return vec


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--fake", type=str, required=True)

    args = parser.parse_args()

    print("Processing real...")
    real_counter = build_distribution(args.real)

    print("Processing fake...")
    fake_counter = build_distribution(args.fake)

    # 统一类别空间
    all_keys = sorted(set(real_counter.keys()) | set(fake_counter.keys()))

    real_prob = counter_to_prob(real_counter, all_keys)
    fake_prob = counter_to_prob(fake_counter, all_keys)

    kl = kl_divergence(real_prob, fake_prob)

    print("\n===== RESULT =====")
    print(f"#classes: {len(all_keys)}")
    print(f"KL(real || fake): {kl:.6f}")

    # 打印分布（前10）
    print("\nTop-10 real:")
    for k, v in real_counter.most_common(10):
        print(f"{k}: {v}")

    print("\nTop-10 fake:")
    for k, v in fake_counter.most_common(10):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
