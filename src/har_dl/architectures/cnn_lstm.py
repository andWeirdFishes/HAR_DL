import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: bool = False,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout: float, negative_slope: float):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7, pool=True, negative_slope=negative_slope),
            nn.Dropout(dropout),
            ConvBlock(32, 64, kernel_size=5, pool=True, negative_slope=negative_slope),
            nn.Dropout(dropout),
            ConvBlock(64, 128, kernel_size=3, negative_slope=negative_slope),
            ConvBlock(128, 128, kernel_size=3, negative_slope=negative_slope),
            nn.Dropout(dropout),
            ConvBlock(128, 256, kernel_size=3, pool=True, negative_slope=negative_slope),
            ConvBlock(256, 256, kernel_size=3, negative_slope=negative_slope),
            nn.Dropout(dropout),
            ConvBlock(256, 512, kernel_size=3, negative_slope=negative_slope),
            ConvBlock(512, 512, kernel_size=3, negative_slope=negative_slope),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_head: float,
        dropout_mid: float,
        negative_slope: float,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(in_features, 256),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout_mid),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout_mid),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout_conv: float = 0.3,
        dropout_head: float = 0.5,
        negative_slope: float = 0.01,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            in_channels=in_channels,
            dropout=dropout_conv,
            negative_slope=negative_slope,
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_conv if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = ClassifierHead(
            in_features=lstm_hidden,
            num_classes=num_classes,
            dropout_head=dropout_head,
            dropout_mid=dropout_conv,
            negative_slope=negative_slope,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)