import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, node_feature_dim, sequence_feature_dim, gat_out_channels=None, gat_heads=None,
                 lstm_hidden_dim=128, lstm_layers=4, edge_dim=None):
        """
        纯LSTM模型（接口与GAT-LSTM兼容，无用参数保留为默认值）
        :param node_feature_dim: 兼容GAT-LSTM的参数（无实际作用）
        :param sequence_feature_dim: 时序序列的特征维度（核心参数）
        :param gat_out_channels: 兼容参数，无作用
        :param gat_heads: 兼容参数，无作用
        :param lstm_hidden_dim: LSTM隐藏层维度
        :param lstm_layers: LSTM层数
        :param edge_dim: 兼容参数，无作用
        """
        super(LSTM, self).__init__()

        # 仅使用时序特征作为输入，无需图特征处理
        self.lstm = nn.LSTM(
            input_size=sequence_feature_dim,  # 仅时序特征维度
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0  # 多层时才用dropout
        )
        self.lstm_dropout = nn.Dropout(0.3)

        # 输出层：预测单步负荷值
        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, sequences, edge_index=None, edge_attr=None, node_features=None, node_indices=None):
        """
        前向传播（保留GAT-LSTM的所有参数位，仅使用sequences）
        :param sequences: 时序序列 [batch_size, seq_len, sequence_feature_dim]
        :param edge_index: 兼容参数，无作用
        :param edge_attr: 兼容参数，无作用
        :param node_features: 兼容参数，无作用
        :param node_indices: 兼容参数，无作用
        :return: 预测值 [batch_size, 1]
        """
        # LSTM处理时序序列
        lstm_out, _ = self.lstm(sequences)
        lstm_out = self.lstm_dropout(lstm_out)

        # 取最后一个时间步的输出做预测
        lstm_out = lstm_out[:, -1, :]

        # 全连接层输出预测结果
        out = self.fc(lstm_out)
        return out