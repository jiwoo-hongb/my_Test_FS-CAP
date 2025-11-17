from torch import nn
import torch.nn.functional as F


class ContextSelfAttention(nn.Module):
    """
    FS-CAP의 context set(r_i들)에 대해 ANP-style self-attention을 적용하는 블록.
    입력:  (batch, n_ctx, d_model)
    출력:  (batch, n_ctx, d_model)  (각 컨텍스트 벡터가 서로를 보고 업데이트됨)
    """
    def __init__(self, d_model, n_heads, n_layers=1, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,  # 입력을 (B, T, C) 형태로 받기 위함
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, n_ctx, d_model)
        return self.encoder(x)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        if config['encoder_batchnorm']:
            self.fc = nn.Sequential(nn.Linear(in_dim, config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['d_model']))
        else:
            self.fc = nn.Sequential(nn.Linear(in_dim, config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['d_model']))

    def forward(self, x, scalar=None):
        if scalar != None:
            return self.fc(x * scalar)
        return self.fc(x)


class Predictor(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        if config['pred_dropout']:
            if config['pred_batchnorm']:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
            else:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
        else:
            if config['pred_batchnorm']:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
            else:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))

    def forward(self, x):
        return self.fc(x)
