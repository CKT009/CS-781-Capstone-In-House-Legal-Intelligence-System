from .decoder import DecoderLM


class EfficientLM(DecoderLM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stack.mode = "sliding"

    @property
    def receptive_field(self):
        return self.cfg.window * self.cfg.n_layers
