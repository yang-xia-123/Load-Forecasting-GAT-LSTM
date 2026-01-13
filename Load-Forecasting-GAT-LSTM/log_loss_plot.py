import matplotlib.pyplot as plt
import numpy as np

def load_loss_data(file_path):
    """读取training_log.txt中的损失数据"""
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line_num, line in enumerate(f, start=2):
                line = line.strip().replace(' ', '')
                if not line:
                    continue
                try:
                    epoch_str, train_loss_str, val_loss_str = line.split(',')
                    epoch = int(epoch_str)
                    train_loss = float(train_loss_str)
                    val_loss = float(val_loss_str)
                    epoch_list.append(epoch)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                except ValueError as e:
                    print(f"警告：第{line_num}行数据错误，跳过 | {e}")
                    continue
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return [], [], []
    return epoch_list, train_loss_list, val_loss_list

# 加载数据
file_path = "outputs/plots/training_log.txt"
epoch, train_loss, val_loss = load_loss_data(file_path)
if not epoch:
    print("数据加载失败！")
else:
    # 调整阶段划分：前15轮（阶段1） + 15轮后（阶段2）
    epoch_np = np.array(epoch)
    stage1_idx = epoch_np <= 15  # 阶段1：前15轮
    stage2_idx = epoch_np > 15   # 阶段2：15轮后

    # 创建2个子图（自动适配真实Y轴）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # -------- 子图1：前15轮（真实Y轴，展示初始快速下降） --------
    ax1.plot(epoch_np[stage1_idx], np.array(train_loss)[stage1_idx],
             label='Train Loss', color='#1f77b4', linewidth=1.5)
    ax1.plot(epoch_np[stage1_idx], np.array(val_loss)[stage1_idx],
             label='Validation Loss', color='#ff7f0e', linewidth=1.5)
    ax1.set_title('Loss Curve (Stage 1: Epoch 1~15)', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # -------- 子图2：15轮后（真实Y轴，展示后期精细下降） --------
    ax2.plot(epoch_np[stage2_idx], np.array(train_loss)[stage2_idx],
             color='#1f77b4', linewidth=1.5)
    ax2.plot(epoch_np[stage2_idx], np.array(val_loss)[stage2_idx],
             color='#ff7f0e', linewidth=1.5)
    ax2.set_title('Loss Curve (Stage 2: Epoch 16~End)', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(alpha=0.3)

    # 自动调整子图间距
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_curve_two_stages_15epoch.png', dpi=300)