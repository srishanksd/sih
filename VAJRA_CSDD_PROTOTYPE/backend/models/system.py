"""High-level VAJRA-CSDD model graph; training integration is intentionally separate."""
from .encoders.fusion import ReliabilityAwareFusion
from .state.convective_state import ConvectiveStateEncoder
from .storm_graph import build_graph
from .probabilistic.ensemble import StochasticEnsemble
from .hazard_heads.multihazard import MultiHazardHeads


class VAJRACSDDModel:
    """Compose the architecture without changing the existing training entry point."""
    def __init__(self, d_model=128, state_dim=128):
        self.fusion = ReliabilityAwareFusion(d_model=d_model)
        self.state = ConvectiveStateEncoder(d_model, state_dim)
        self.ensemble = StochasticEnsemble(state_dim)
        self.hazards = MultiHazardHeads(state_dim)

    def make_storm_graph(self, objects):
        return build_graph(objects)
