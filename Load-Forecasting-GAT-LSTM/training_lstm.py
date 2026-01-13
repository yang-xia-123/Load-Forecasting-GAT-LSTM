import torch
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from lstm_model import LSTM  # 替换为纯LSTM模型
from data_preprocessing import preprocess_data, load_config

# 配置日志
logging.basicConfig(level=logging.INFO)


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=200, patience=10,
                output_dir="outputs_lstm"):
    """复用原GAT-LSTM的训练逻辑，无修改"""
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    trigger_times = 0

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "lstm_model.pth")
    training_log_path = os.path.join(output_dir, "training_log.txt")

    with open(training_log_path, 'w') as log_file:
        log_file.write("Epoch, Train Loss, Validation Loss\n")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for sequences, targets, nodes in train_loader:
                sequences, targets, nodes = sequences.to(device), targets.to(device), nodes.to(device)
                optimizer.zero_grad()
                # 调用LSTM模型（忽略图相关参数）
                output = model(sequences)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets, nodes in val_loader:
                    sequences, targets, nodes = sequences.to(device), targets.to(device), nodes.to(device)
                    output = model(sequences)
                    loss = criterion(output, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            log_msg = f"Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}"
            logging.info(log_msg)
            log_file.write(f"{epoch + 1},{train_loss},{val_loss}\n")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best LSTM model at {model_save_path}")
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info("Early stopping triggered!")
                    break

    # 保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    logging.info(f"Loss curve saved at {loss_curve_path}")

    return train_losses, val_losses


if __name__ == "__main__":
    # 加载配置和预处理数据（完全复用原逻辑）
    config = load_config()
    train_seq, train_tgt, train_nodes, val_seq, val_tgt, val_nodes, test_seq, test_tgt, test_nodes, node_features_tensor, edge_index_tensor, edge_attr_tensor, target_scaler = preprocess_data(
        config)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_seq, train_tgt, train_nodes = train_seq.to(device), train_tgt.to(device), train_nodes.to(device)
    val_seq, val_tgt, val_nodes = val_seq.to(device), val_tgt.to(device), val_nodes.to(device)

    # 准备DataLoader（复用原逻辑）
    batch_size = 27
    train_dataset = TensorDataset(train_seq, train_tgt, train_nodes)
    val_dataset = TensorDataset(val_seq, val_tgt, val_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型参数（与GAT-LSTM的LSTM部分保持一致，确保公平对比）
    sequence_feature_dim = train_seq.shape[2]
    lstm_hidden_dim = 128
    lstm_layers = 4

    # 初始化纯LSTM模型
    model = LSTM(
        node_feature_dim=0,  # 无实际作用
        sequence_feature_dim=sequence_feature_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers
    ).to(device)

    # 优化器、调度器、损失函数（与GAT-LSTM保持一致）
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = torch.nn.MSELoss()

    # 训练模型
    output_dir = os.path.join(config['output_dir'], "lstm_results")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=200,
        patience=10,
        output_dir=output_dir
    )

    logging.info("LSTM Training completed successfully.")