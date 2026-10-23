"""Stand-in for comfy.ldm.modules.attention."""

import torch


class AttentionTensorContainer:
    """Same single-owner wrapper the real ComfyUI uses."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        return self.tensor

    def take(self):
        tensor = self.tensor
        self.tensor = None
        return tensor


CALLS = {"optimized": 0}


def optimized_attention(q, k, v, heads, mask=None, attn_precision=None, transformer_options=None, **kwargs):
    """Counts pass-through calls and runs plain attention."""
    CALLS["optimized"] += 1
    q = q.take() if isinstance(q, AttentionTensorContainer) else q
    k = k.take() if isinstance(k, AttentionTensorContainer) else k
    v = v.take() if isinstance(v, AttentionTensorContainer) else v
    batch, q_len, inner = q.shape
    dim = inner // heads
    shape = lambda t: t.view(t.shape[0], t.shape[1], heads, dim).transpose(1, 2)
    qh, kh, vh = shape(q), shape(k), shape(v)
    probs = torch.softmax(torch.matmul(qh, kh.transpose(-2, -1)) / dim**0.5, dim=-1)
    return torch.matmul(probs, vh).transpose(1, 2).reshape(batch, q_len, inner)
