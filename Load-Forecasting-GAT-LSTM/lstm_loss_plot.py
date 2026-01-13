import matplotlib.pyplot as plt
import numpy as np


def load_loss_data(file_path):
    """读取training_log.txt中的LSTM损失数据（Epoch, Train Loss, Validation Loss）"""
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头行
            for line_num, line in enumerate(f, start=2):
                line = line.strip().replace(' ', '')  # 去除空格和换行符
                if not line:  # 跳过空行
                    continue
                try:
                    # 分割并转换为数值类型
                    epoch_str, train_loss_str, val_loss_str = line.split(',')
                    epoch = int(epoch_str)
                    train_loss = float(train_loss_str)
                    val_loss = float(val_loss_str)

                    epoch_list.append(epoch)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                except ValueError as e:
                    print(f"警告：第{line_num}行数据格式错误，跳过 | 错误：{e}")
                    continue
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}，请检查文件路径是否正确！")
        return [], [], []
    return epoch_list, train_loss_list, val_loss_list


# -------------------------- 核心绘图逻辑 --------------------------
# 加载数据（文件和代码同目录时直接用'training_log.txt'）
file_path = "outputs/plots/lstm_results/training_log.txt"
epoch, train_loss, val_loss = load_loss_data(file_path)

# 检查数据加载是否成功
if not epoch or not train_loss or not val_loss:
    print("数据加载失败，无法绘制图表！")
else:
    # 转换为numpy数组，方便阶段筛选
    epoch_np = np.array(epoch)
    # 阶段划分：前15轮（阶段1）、15轮后（阶段2）
    stage1_idx = epoch_np <= 15  # 前15轮（快速下降阶段）
    stage2_idx = epoch_np > 15  # 15轮后（精细下降阶段）

    # 创建2个子图（垂直排列，尺寸适配可读性）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # -------- 子图1：前15轮（真实Y轴，展示初始快速下降） --------
    ax1.plot(epoch_np[stage1_idx], np.array(train_loss)[stage1_idx],
             label='Train Loss', color='#1f77b4', linewidth=1.5)
    ax1.plot(epoch_np[stage1_idx], np.array(val_loss)[stage1_idx],
             label='Validation Loss', color='#ff7f0e', linewidth=1.5)
    ax1.set_title('LSTM Loss Curve (Stage 1: Epoch 1~15)', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right')  # 图例位置
    ax1.grid(alpha=0.3)  # 浅灰色网格，提升可读性

    # -------- 子图2：15轮后（真实Y轴，展示后期精细下降） --------
    ax2.plot(epoch_np[stage2_idx], np.array(train_loss)[stage2_idx],
             color='#1f77b4', linewidth=1.5)
    ax2.plot(epoch_np[stage2_idx], np.array(val_loss)[stage2_idx],
             color='#ff7f0e', linewidth=1.5)
    ax2.set_title('LSTM Loss Curve (Stage 2: Epoch 16~End)', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(alpha=0.3)

    # 自动调整子图间距，避免标题/标签重叠
    plt.tight_layout()
    # 显示图表（如需保存，取消下方注释）
    plt.show()
    fig.savefig('lstm_loss_curve_two_stages.png', dpi=300, bbox_inches='tight')