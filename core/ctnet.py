import math
import torch
from torch import nn
from dropblock import DropBlock2D

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNNTransformerRegressor(nn.Module):
    """
    架构：EEGNet-style CNN (特征提取) + Transformer (时序建模) + Regression Head
    特点：时空分离卷积，参数少，抗过拟合，适配 KF。
    """
    def __init__(self, in_channels=8, output_dims=6, cnn_out_channels=32,
                 d_model=None, nhead=2, transformer_layers=2, dropout_rate=0.5): # 建议 dropout 提高到 0.5
        super().__init__()

        # 如果未指定 d_model，默认设为 cnn_out_channels (推荐 32 或 64)
        if d_model is None:
            d_model = cnn_out_channels
        self.d_model = d_model

        # --- 1. 改进后的 CNN 特征提取器 (时空分离) ---
        # 这种结构强迫模型先学频率，再学空间协同，极大降低过拟合风险
        self.cnn_extractor = nn.Sequential(
            # [Step 1] 时域卷积 (Temporal Conv)
            # 只卷时间 (1, 64)，不卷通道。相当于给每个电极做带通滤波。
            # 输出: (B, 16, C, T)
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(), # 生物信号推荐用 ELU
            
            # [Step 2] 空域卷积 (Spatial Conv) - 核心修改
            # kernel_size=(in_channels, 1) -> 一次性看全所有电极！
            # 无论电极怎么排，这里都能学到全局关系。
            # groups=16 -> Depthwise Conv，每个时域特征图独立学习空间模式
            # 输出: (B, 32, 1, T) -> 空间维度被压缩为 1
            nn.Conv2d(16, cnn_out_channels, kernel_size=(in_channels, 1), 
                      groups=16, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ELU(),
            
            # 第一次降采样 & DropBlock
            nn.AvgPool2d(kernel_size=(1, 4)),
            DropBlock2D(block_size=5, drop_prob=dropout_rate), # 强力抗过拟合

            # [Step 3] 特征整合 (Separable Conv)
            # 进一步提取高层特征并降采样
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=(1, 16), 
                      padding=(0, 8), groups=cnn_out_channels, bias=False),
            nn.Conv2d(cnn_out_channels, d_model, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            
            # 第二次降采样
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate), # 标准 Dropout

            # 强制对齐时间维度，方便 Transformer 处理
            # 输出形状: (B, d_model, 1, 40) -> 40 是序列长度
            nn.AdaptiveAvgPool2d((1, 24)) 
        )

        # --- 2. Transformer (时序建模) ---
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 2, # 稍微减小前馈层参数
            dropout=0.2, # Transformer 内部 dropout 不宜过高
            batch_first=True
        )
        # 减少层数以防止过拟合 (推荐 2 层)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # --- 3. 回归头 ---
        # 增加一个 Projection 层，帮助 KF 更好地进行线性拟合
        self.kf_projection = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.Tanh() # Tanh 均值为 0，利于 KF
        )
        self.regression_head = nn.Linear(64, output_dims)

    def forward(self, x, return_feature=False):
        """
        Args:
            x: 输入张量 (Batch, Channels, Time)
            return_feature: 是否返回倒数第二层特征 (用于 KF 训练)
        """
        B, C, L = x.shape
        
        # 1. 增加 Channel 维度: (B, 1, C, T)
        x = x.unsqueeze(1) 
        
        # 2. CNN 特征提取
        # Output: (B, d_model, 1, Seq_Len=40)
        cnn_features = self.cnn_extractor(x) 

        # 3. 维度变换适配 Transformer
        # (B, d_model, 1, T) -> (B, d_model, T) -> (B, T, d_model)
        x = cnn_features.squeeze(2).permute(0, 2, 1) 

        # 4. Transformer 编码
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x) 
        
        # 5. 取最后一个时间步 (Last Token) 代表当前状态
        # (B, T, d_model) -> (B, d_model)
        last_token = transformer_output.mean(dim=1)
        
        # 6. 经过投影层 (这是给 KF 的最佳特征)
        feature = self.kf_projection(last_token)

        # --- 关键：支持返回特征给 KF 训练 ---
        if return_feature:
            return feature
        # --------------------------------

        # 7. 最终回归
        output = self.regression_head(feature) 

        return output
