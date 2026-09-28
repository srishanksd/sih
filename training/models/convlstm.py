import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad)

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()
        g = g.tanh()
        c = f * c + i * g
        h = o * c.tanh()
        return h, c

    def init_state(self, x):
        shape = (x.size(0), self.hidden_channels, x.size(2), x.size(3))
        return x.new_zeros(shape), x.new_zeros(shape)


class ConvLSTMNowcaster(nn.Module):
    """Autoregressive ConvLSTM baseline for radar nowcasting.

    Input:  [B, T_in, C, H, W]
    Output: [B, T_out, C_out, H, W]
    """
    def __init__(self, in_channels=1, hidden_channels=(32, 64), out_channels=1, kernel_size=3):
        super().__init__()
        self.cells = nn.ModuleList()
        channels = in_channels
        for hidden in hidden_channels:
            self.cells.append(ConvLSTMCell(channels, hidden, kernel_size))
            channels = hidden
        self.head = nn.Conv2d(channels, out_channels, 1)

    def _encode_step(self, x, states):
        new_states = []
        for cell, state in zip(self.cells, states):
            h, c = cell(x, state)
            new_states.append((h, c))
            x = h
        return x, new_states

    def forward(self, x, future_steps):
        states = [cell.init_state(x[:, 0]) for cell in self.cells]
        for t in range(x.size(1)):
            hidden, states = self._encode_step(x[:, t], states)
        outputs = []
        current = x[:, -1]
        for _ in range(future_steps):
            hidden, states = self._encode_step(current, states)
            current = self.head(hidden).sigmoid()
            outputs.append(current)
        return torch.stack(outputs, dim=1)
