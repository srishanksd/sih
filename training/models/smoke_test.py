import torch
from models.convlstm import ConvLSTMNowcaster

if __name__ == "__main__":
    model = ConvLSTMNowcaster()
    x = torch.zeros(2, 8, 1, 32, 32)
    y = model(x, future_steps=4)
    assert y.shape == (2, 4, 1, 32, 32)
    print("ConvLSTM smoke test passed:", tuple(y.shape))