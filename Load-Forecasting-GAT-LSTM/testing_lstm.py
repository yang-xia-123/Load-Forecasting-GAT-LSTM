import torch
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTM  # 替换为纯LSTM模型
from data_preprocessing import preprocess_data, load_config

# 配置日志
logging.basicConfig(level=logging.INFO)

def evaluate_model(model, test_loader, target_scaler, node_to_state, test_time_indices, output_dir="outputs_lstm"):
    """复用原评估逻辑，仅修改模型调用方式"""
    model.eval()
    test_predictions, test_targets, test_nodes, test_hours = [], [], [], []

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")

    with torch.no_grad():
        for i, (sequences, targets, nodes) in enumerate(test_loader):
            sequences, targets, nodes = sequences.to(device), targets.to(device), nodes.to(device)
            # 调用LSTM模型（忽略图相关参数）
            output = model(sequences)
            test_predictions.append(output.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
            test_nodes.append(nodes.cpu().numpy())

            batch_size = targets.shape[0]
            batch_time_indices = test_time_indices[i * batch_size:(i + 1) * batch_size]
            test_hours.extend(batch_time_indices % 24)

    # 拼接结果（复用原逻辑）
    test_predictions = np.concatenate(test_predictions).squeeze()
    test_targets = np.concatenate(test_targets).squeeze()
    test_nodes = np.concatenate(test_nodes).squeeze()

    # 逆标准化（复用原逻辑）
    test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

    # 保存预测结果（复用原逻辑）
    results_df = pd.DataFrame({
        "Node": [node_to_state.get(node, f"Node {node}") for node in test_nodes],
        "Hour": test_hours,
        "Actual": test_targets,
        "Predicted": test_predictions
    })
    results_df.to_csv(predictions_path, index=False)
    logging.info(f"Saved predictions and actuals to {predictions_path}")

    # 计算评估指标（复用原逻辑）
    mae = mean_absolute_error(test_targets, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    mape = np.mean(np.abs((test_targets - test_predictions) / np.clip(test_targets, a_min=1e-8, a_max=None))) * 100
    r2 = r2_score(test_targets, test_predictions)
    corr_coef, _ = pearsonr(test_targets, test_predictions)

    # 保存指标（复用原逻辑）
    with open(evaluation_metrics_path, "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape}%\n")
        f.write(f"R-squared (R2): {r2}\n")
        f.write(f"Pearson Correlation Coefficient: {corr_coef}\n")
    logging.info(f"Saved evaluation metrics to {evaluation_metrics_path}")

    # 绘制24小时均值对比图（复用原逻辑）
    plt.figure(figsize=(12, 6))
    all_actuals, all_predicteds = [], []
    unique_nodes = np.unique(test_nodes)

    for node in unique_nodes:
        node_indices = np.where(test_nodes == node)[0][:24]
        all_actuals.append(test_targets[node_indices])
        all_predicteds.append(test_predictions[node_indices])

    mean_actual = np.mean(np.array(all_actuals), axis=0)
    mean_predicted = np.mean(np.array(all_predicteds), axis=0)

    plt.plot(mean_actual, label='Mean Actual', alpha=0.8)
    plt.plot(mean_predicted, label='Mean Predicted', alpha=0.8)
    plt.title('LSTM - Mean Actual vs Predicted Values Over 24-Hour Period (All Nodes)')
    plt.xlabel('Hour')
    plt.ylabel('Load')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted_24hr_all_nodes.png"), dpi=300)
    plt.close()

    # 绘制单节点24小时对比图（复用原逻辑）
    plt.figure(figsize=(30, 25))
    rows, cols = 5, 6
    for i, node in enumerate(unique_nodes[:rows * cols]):
        node_indices = np.where(test_nodes == node)[0][:24]
        actual = test_targets[node_indices]
        predicted = test_predictions[node_indices]

        plt.subplot(rows, cols, i + 1)
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        node_name = node_to_state.get(node, f"Node {node}")
        plt.title(f"LSTM - Actual vs Predicted - {node_name}")
        plt.xlabel('Hour')
        plt.ylabel('Load')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted_24hr_individual_nodes.png"), dpi=300)
    plt.close()
    logging.info(f"Saved LSTM Actual vs Predicted plots to {output_dir}")

    return test_targets, test_predictions, mae, rmse, mape, r2, corr_coef

if __name__ == "__main__":
    # 加载配置和数据（复用原逻辑）
    config = load_config()
    train_seq, train_tgt, train_nodes, val_seq, val_tgt, val_nodes, test_seq, test_tgt, test_nodes, node_features_tensor, edge_index_tensor, edge_attr_tensor, target_scaler = preprocess_data(config)

    test_time_indices = np.arange(len(test_tgt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化LSTM模型（与训练时参数一致）
    model = LSTM(
        node_feature_dim=0,
        sequence_feature_dim=test_seq.shape[2],
        lstm_hidden_dim=128,
        lstm_layers=4
    ).to(device)

    # 加载训练好的LSTM模型
    model_path = os.path.join(config['output_dir'], "lstm_results", "lstm_model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    logging.info(f"Loaded LSTM model from {model_path}")

    # 数据加载（复用原逻辑）
    test_seq, test_tgt, test_nodes = test_seq.to(device), test_tgt.to(device), test_nodes.to(device)
    batch_size = 27
    test_dataset = TensorDataset(test_seq, test_tgt, test_nodes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 节点映射（复用原逻辑）
    node_mapping = {node: idx for idx, node in enumerate(sorted(np.unique(test_nodes.cpu())))}
    node_to_state = {idx: node for node, idx in node_mapping.items()}

    # 评估模型
    output_dir = os.path.join(config['output_dir'], "lstm_results")
    test_targets, test_predictions, mae, rmse, mape, r2, corr_coef = evaluate_model(
        model=model,
        test_loader=test_loader,
        target_scaler=target_scaler,
        node_to_state=node_to_state,
        test_time_indices=test_time_indices,
        output_dir=output_dir
    )

    logging.info("LSTM Testing completed successfully.")