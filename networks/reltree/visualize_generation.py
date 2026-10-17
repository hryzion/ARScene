"""Draw a static, cumulative layer replay from saved generation tokens."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Polygon
import numpy as np

from .structures import NodeType
from .tokenizer import EncodedTree, RelTreeTokenizer


CATEGORY_NAMES = {
    "pendant_lamp": "吊灯", "ceiling_lamp": "吸顶灯", "bookshelf": "书架",
    "dining_chair": "餐椅", "dining_table": "餐桌", "coffee_table": "茶几",
    "loveseat_sofa": "双人沙发", "multi_seat_sofa": "多人沙发",
    "l_shaped_sofa": "L 型沙发", "tv_stand": "电视柜", "armchair": "扶手椅",
    "cabinet": "柜子", "console_table": "玄关桌", "round_end_table": "圆边桌",
    "corner_side_table": "边桌", "shelf": "置物架", "desk": "书桌",
    "stool": "凳子", "wardrobe": "衣柜", "wine_cabinet": "酒柜",
    "lounge_chair": "休闲椅", "lazy_sofa": "懒人沙发",
}


def load_generation(sample_path, tokenizer_path):
    record = json.loads(sample_path.read_text())
    if record.get("status") != "complete" or not record.get("tokens"):
        raise ValueError("A complete generated scene with saved tokens is required")
    tokenizer = RelTreeTokenizer.load(tokenizer_path)
    encoded = EncodedTree.from_dict(record["tokens"])
    tree = tokenizer.decode(encoded)
    rows = list(tree.breadth_first())
    if len(rows) != len(record["history"]):
        raise ValueError("Token/history lengths differ")
    for index, ((node, parent, depth, slot, path), saved) in enumerate(zip(rows, record["history"])):
        actual = (int(node.kind), parent, depth, slot, list(path))
        expected = tuple(saved[key] for key in ("kind", "parent", "depth", "slot", "path"))
        if actual != expected:
            raise ValueError(f"Generation history differs from decoded tree at node {index}")
        np.testing.assert_array_equal(encoded.geometry[index], saved["geometry"])
        assert encoded.classes[index] == saved["category"]
    if tree.root.box.to_dict() != record["room"]:
        raise ValueError("Saved Room differs from decoded Room")
    if len(tree.objects) != len(record["objects"]):
        raise ValueError("Object counts differ")
    for actual, saved in zip(tree.objects, record["objects"]):
        assert actual.category == saved["category"]
        for key, value in actual.box.to_dict().items():
            np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-10)
    return record, tree, rows


def draw_layers(record, tree, rows, output):
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    max_depth = max(row[2] for row in rows)
    nrows = (max_depth + 3) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 3.6 * nrows), squeeze=False)
    fig.patch.set_facecolor("#f6f8fb")
    points = np.concatenate([node.box.footprint() for node, *_ in rows if node.box is not None])
    lo, hi = points.min(0) - .45, points.max(0) + .45
    room = tree.root.box.footprint()
    categories = list(dict.fromkeys(obj.category for obj in tree.objects))
    palette = plt.get_cmap("tab20")
    colors = {category: palette(2 * i % 20) for i, category in enumerate(categories)}
    object_ids = {index: f"O{node.object_index + 1:02d}"
                  for index, (node, *_) in enumerate(rows) if node.kind == NodeType.OBJECT}
    states = []
    for depth, ax in zip(range(max_depth + 1), axes.flat):
        ax.set_facecolor("#ffffff")
        ax.add_patch(Polygon(room, facecolor="#f0f3f6", edgecolor="#475569", lw=1.6, zorder=0))
        visible_objects, frontier = [], []
        # A completed layer expands the previous SPLIT frontier. Objects persist.
        for index, (node, _, node_depth, _, _) in enumerate(rows):
            if node.kind == NodeType.OBJECT and node_depth <= depth:
                visible_objects.append(index)
            elif node.kind == NodeType.SPLIT and node_depth == depth:
                frontier.append(index)
        for index in frontier:
            box = rows[index][0].box
            ax.add_patch(Polygon(box.footprint(), facecolor=(.145, .388, .922, .07), edgecolor="#2563eb",
                                 lw=1.8, linestyle=(0, (5, 3)), zorder=1))
            anchor = box.footprint().min(0)
            ax.annotate(f"S{index}", anchor, xytext=(3, 4), textcoords="offset points",
                        color="#1d4ed8", fontsize=9, fontweight="bold", zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
        for index in visible_objects:
            node, _, node_depth, _, _ = rows[index]
            category = tree.objects[node.object_index].category
            is_new = node_depth == depth
            ax.add_patch(Polygon(node.box.footprint(), facecolor=colors[category], alpha=.57,
                                 edgecolor="#9a5800" if is_new else "#475569",
                                 linewidth=2.4 if is_new else .9, zorder=3))
            # Identical IDs across panels allow tracking objects born at earlier depths.
            ax.annotate(object_ids[index], (node.box.position[0], node.box.position[2]),
                    xytext=(-9, 12) if category in ("pendant_lamp", "ceiling_lamp") else (0, 0),
                    textcoords="offset points",
                    color="#7c4200" if is_new else "#243449", ha="center", va="center",
                    fontsize=8, fontweight="bold" if is_new else "normal", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .8})
        current = Counter(node.kind for node, _, d, _, _ in rows if d == depth)
        name = "起点 · 生成 Room" if depth == 0 else f"第 {depth} 层" + (" · 生成结束" if depth == max_depth else "")
        caption = (f"房间尺寸 {tree.root.box.size[0]:.2f} × {tree.root.box.size[2]:.2f} m，高 {tree.root.box.size[1]:.2f} m"
                   if depth == 0 else
                   f"新增 {current[NodeType.SPLIT]} 个 SPLIT、{current[NodeType.OBJECT]} 个物体"
                   f"  |  累计 {len(visible_objects)} 个物体")
        ax.set_title(f"{name}  /  depth={depth}\n{caption}", loc="left", fontsize=11, pad=11)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x / m", ylabel="z / m")
        ax.set_aspect("equal")
        ax.grid(color="#d9e0e9", lw=.5, alpha=.5)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d4dce6")
        states.append({"depth": depth, "new_splits": current[NodeType.SPLIT],
                       "new_objects": current[NodeType.OBJECT], "cumulative_objects": len(visible_objects),
                       "frontier_node_ids": frontier})
    legend_ax = axes.flat[max_depth + 1]
    legend_ax.axis("off")
    legend_ax.text(0, .98, "如何阅读", fontsize=14, weight="bold", va="top", transform=legend_ax.transAxes)
    legend_ax.legend(handles=[
        Patch(facecolor="#f0f3f6", edgecolor="#475569", label="灰色轮廓：模型生成的 Room"),
        Patch(facecolor="#dbeafe", edgecolor="#2563eb", linestyle="--", label="蓝色虚线 S：下一层继续展开的 SPLIT"),
        Patch(facecolor="#8bc5a5", edgecolor="#475569", label="彩色实框 O：已生成物体，之后保持不变"),
        Patch(facecolor="#8bc5a5", edgecolor="#9a5800", lw=2.4, label="棕色粗边：本层新增物体"),
    ], loc="upper left", bbox_to_anchor=(-.02, .90), frameon=False, fontsize=10)
    legend_ax.text(0, .36, "物体编号（各层一致）", fontsize=11, weight="bold", transform=legend_ax.transAxes)
    grouped = []
    for category in categories:
        ids = [object_ids[i] for i in object_ids if tree.objects[rows[i][0].object_index].category == category]
        grouped.append((category, ", ".join(ids)))
    legend_rows = max(4, (len(grouped) + 1) // 2)
    for i, (category, ids) in enumerate(grouped):
        column, row = divmod(i, legend_rows)
        legend_ax.text(column * .53, .26 - row * (.32 / legend_rows),
                       f"{ids}  {CATEGORY_NAMES.get(category, category)}", fontsize=9 if legend_rows <= 4 else 8.5,
                       color=colors[category], transform=legend_ax.transAxes)
    for extra in list(axes.flat)[max_depth + 2:]:
        extra.axis("off")
    fig.suptitle(f"RelTree · Livingroom 逐层生成\n{record['sample_id']}  |  last.pt · step {record['checkpoint_step']:,}"
                 f"  |  {len(tree.objects)} 个物体 · {max_depth} 层展开", x=.05, y=.986,
                 ha="left", fontsize=18, weight="bold", color="#172b43")
    fig.text(.05, .027, "俯视图（x–z），所有面板使用相同坐标尺度；展示每层生成完成后的累计状态。",
             fontsize=10, color="#475569")
    fig.text(.05, .012, "来源：保存的无条件采样历史，未重新建树。图中为 OBB 投影；同一层内部仍按节点和属性自回归生成。",
             fontsize=10, color="#475569")
    fig.subplots_adjust(left=.065, right=.97, bottom=.07, top=.90, hspace=.48, wspace=.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record, tree, rows = load_generation(args.sample, args.tokenizer)
    states = draw_layers(record, tree, rows, args.output)
    print(json.dumps({"output": str(args.output), "sample_id": record["sample_id"],
                      "checked_history_and_decoded_objects": True, "layers": states}, ensure_ascii=False))


if __name__ == "__main__":
    main()
