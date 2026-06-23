import torch
from har_dl.architectures.cnn_lstm_absmax import CNNLSTMAbsMax 

from har_dl.definitions import get_project_root

import os

save_path = os.path.join(get_project_root(),'artifacts','model_architecture.onnx')

model = CNNLSTMAbsMax(in_channels=8)
model.eval()
dummy_input = torch.randn(1, 8, 100)

torch.onnx.export(
    model,
    dummy_input,
    save_path,
    input_names=["input_signal"],
    output_names=["class_logits"],
    opset_version=14,
)