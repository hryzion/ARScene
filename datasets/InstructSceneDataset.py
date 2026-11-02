import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.SceneTokenNormalizer import SceneTokenNormalizer


class ThreeDFrontInstructSceneDataset(Dataset):
    """ARScene processed data adapted to InstructScene's Sg2ScDiffusion.

    Expected per-scene files:
      - room_data.npz with obj_tokens [N, C + 7]
      - relations.npy with rows [subject_index, predicate_index, object_index]
    """

    def __init__(
        self,
        dataset_dir,
        split="train",
        num_classes=31,
        padded_length=27,
        num_predicates=10,
        use_normalizer=True,
    ):
        self.root = os.path.join(dataset_dir, split)
        self.num_classes = int(num_classes)
        self.padded_length = int(padded_length)
        self.num_predicates = int(num_predicates)
        self.use_normalizer = use_normalizer
        self.file_list = sorted(
            f for f in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, f))
        )
        self.normalizer = None
        normalizer_path = os.path.join(dataset_dir, "normalizer_stats_atiss.json")
        if use_normalizer and os.path.exists(normalizer_path):
            self.normalizer = SceneTokenNormalizer(
                category_dim=self.num_classes + 2,
                rotation_mode="sincos",
                use_objlat=False,
                atiss=True,
            )
            self.normalizer.load(normalizer_path)

    def __len__(self):
        return len(self.file_list)

    def _load_relations(self, scene_dir, n_objects):
        relations_path = os.path.join(scene_dir, "relations.npy")
        edges = torch.full(
            (self.padded_length, self.padded_length),
            self.num_predicates,
            dtype=torch.long,
        )
        edges_tril = torch.full(
            (self.padded_length * (self.padded_length - 1) // 2,),
            self.num_predicates,
            dtype=torch.long,
        )
        if not os.path.exists(relations_path):
            return edges, edges_tril

        relations = np.load(relations_path, allow_pickle=True)
        if relations.size == 0:
            return edges, edges_tril
        relations = np.asarray(relations, dtype=np.int64).reshape(-1, 3)
        for subject, predicate, obj in relations:
            if (
                0 <= subject < n_objects
                and 0 <= obj < n_objects
                and subject != obj
                and 0 <= predicate < self.num_predicates
            ):
                subject = int(subject)
                obj = int(obj)
                predicate = int(predicate)
                reverse_predicate = predicate
                if self.num_predicates % 2 == 0:
                    half = self.num_predicates // 2
                    reverse_predicate = predicate + half if predicate < half else predicate - half
                edges[subject, obj] = predicate
                edges[obj, subject] = reverse_predicate
                i, j = sorted((subject, obj))
                tri_idx = i * self.padded_length + j - ((i + 2) * (i + 1)) // 2
                edges_tril[tri_idx] = predicate if subject < obj else reverse_predicate
        return edges, edges_tril

    def __getitem__(self, idx):
        scene_name = self.file_list[idx]
        scene_dir = os.path.join(self.root, scene_name)
        data = np.load(os.path.join(scene_dir, "room_data.npz"), allow_pickle=True)
        obj_tokens = torch.from_numpy(data["obj_tokens"].astype(np.float32))
        if self.normalizer is not None:
            obj_tokens = self.normalizer.transform_atiss(
                obj_tokens,
                normalize_rotation=False,
            )

        if obj_tokens.shape[0] > self.padded_length:
            raise ValueError(
                f"Scene {scene_name} has {obj_tokens.shape[0]} objects, "
                f"exceeding padded_length={self.padded_length}"
            )

        n_objects = obj_tokens.shape[0]
        objs = torch.full((self.padded_length,), self.num_classes, dtype=torch.long)
        boxes = torch.zeros((self.padded_length, 8), dtype=torch.float32)
        obj_masks = torch.zeros((self.padded_length,), dtype=torch.long)
        objfeat_vq_indices = torch.full((self.padded_length, 4), 64, dtype=torch.long)

        class_labels = obj_tokens[:, :self.num_classes]
        objs[:n_objects] = torch.argmax(class_labels, dim=-1).long()
        objfeat_vq_indices[:n_objects] = 0
        geom_start = self.num_classes + 2
        translations = obj_tokens[:, geom_start:geom_start + 3]
        sizes = obj_tokens[:, geom_start + 3:geom_start + 6]
        angles = obj_tokens[:, geom_start + 6:geom_start + 7]
        boxes[:n_objects] = torch.cat(
            [translations, sizes, torch.cos(angles), torch.sin(angles)], dim=-1
        )
        obj_masks[:n_objects] = 1
        edges, edges_tril = self._load_relations(scene_dir, n_objects)

        return {
            "room_name": scene_name,
            "description": str(data["description"]),
            "objs": objs,
            "edges": edges,
            "edges_tril": edges_tril,
            "boxes": boxes,
            "obj_masks": obj_masks,
            "objfeat_vq_indices": objfeat_vq_indices,
        }

    @staticmethod
    def collate_fn(samples):
        batch = {}
        batch["room_name"] = [sample["room_name"] for sample in samples]
        batch["description"] = [sample["description"] for sample in samples]
        for key in ("objs", "edges", "edges_tril", "boxes", "obj_masks", "objfeat_vq_indices"):
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
        return batch

    @staticmethod
    def collate_fn_sg2sc(samples):
        max_length = max(int(sample["obj_masks"].sum().item()) for sample in samples)
        batch = {
            "room_name": [sample["room_name"] for sample in samples],
            "description": [sample["description"] for sample in samples],
            "objs": [],
            "edges": [],
            "edges_tril": [],
            "boxes": [],
            "obj_masks": [],
            "objfeat_vq_indices": [],
        }

        for sample in samples:
            length = int(sample["obj_masks"].sum().item())
            batch["objs"].append(sample["objs"][:max_length])
            batch["edges"].append(sample["edges"][:max_length, :max_length])
            batch["boxes"].append(sample["boxes"][:max_length])
            batch["obj_masks"].append(sample["obj_masks"][:max_length])
            batch["objfeat_vq_indices"].append(sample["objfeat_vq_indices"][:max_length])

            edges_tril = torch.full(
                (max_length * (max_length - 1) // 2,),
                int(sample["edges"].max().item()),
                dtype=sample["edges_tril"].dtype,
            )
            cursor = 0
            for i in range(max_length):
                for j in range(i + 1, max_length):
                    edges_tril[cursor] = sample["edges"][i, j]
                    cursor += 1
            batch["edges_tril"].append(edges_tril)

        for key in ("objs", "edges", "edges_tril", "boxes", "obj_masks", "objfeat_vq_indices"):
            batch[key] = torch.stack(batch[key], dim=0)
        return batch
