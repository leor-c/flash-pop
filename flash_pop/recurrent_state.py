from typing import Optional, Union
from torch import Tensor, LongTensor


class RecurrentState:
    def __init__(self, state: Optional[Tensor] = None, index: Optional[Union[int, LongTensor]] = None):
        """
        :param state: a Tensor of shape (num layers, batch, num heads, head_dim_v, head_dim_v)
        :param index: The time step index of the state, corresponding to the number of tokens
        already processed.
        """
        self.state = state
        self.index = index

    def to_dict(self):
        return {'state': self.state, 'index': self.index}

    def cpu_clone(self):
        index = self.index if isinstance(self.index, int) else self.index.clone().detach().cpu()
        return RecurrentState(self.state.clone().detach().cpu(), index)

    def clone(self):
        state = self.state.clone() if self.state is not None else None
        index = self.index.clone() if isinstance(self.index, Tensor) else self.index
        return RecurrentState(state, index)

    def to_device(self, device):
        if self.state is not None:
            self.state = self.state.to(device)
        if isinstance(self.index, LongTensor):
            self.index = self.index.to(device)
        else:
            assert isinstance(self.index, int)

        return self
