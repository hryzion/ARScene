import matplotlib.pyplot as plt

def plot_loss_curve(loss_list, save_path=None):
    plt.figure(figsize=(10, 2))

    x = list(range(1, len(loss_list) + 1))

    plt.plot(
        x,
        loss_list,
        marker='^',
        linewidth=2,
        color='purple',
        markerfacecolor='none',
        markeredgewidth=1.5
    )

    # 标注数值
    offset = (max(loss_list) - min(loss_list) + 1e-8) * 0.03
    for xi, yi in zip(x, loss_list):
        plt.text(
            xi, yi + offset,
            f"{yi:.4f}",
            fontsize=10,
            ha='center'
        )

    # 美化
    # plt.title("Loss Curve", fontsize=14)
    # plt.ylabel("Loss", fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    ax = plt.gca()

    # ❌ 去掉横坐标
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)

    # 去掉右上边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

loss = [0.2240, 0.0089, 0.0035, 0]
plot_loss_curve(loss, save_path="loss_curve.png")