"""DAAM-Mixed: attention attribution maps that stop at the last step.

A merge of the two public DAAM packs, written against the current ComfyUI API:
from the newer pack comes the sampling side (a drop-in sampler, accumulation in
place, no monkeypatched kernels, maps at their native image/16 grid), from the
older one the step window and the pixel level mask work.
"""

from .nodes.explorer import DAAMExplorer
from .nodes.guide import DAAMGuide, DAAMMerge
from .nodes.mask import DAAMMaskCleanup, DAAMMasks
from .nodes.sampler import DAAMKSampler, DAAMSamplerCustom

NODE_CLASS_MAPPINGS = {
    "DAAMKSampler": DAAMKSampler,
    "DAAMSamplerCustom": DAAMSamplerCustom,
    "DAAMMasks": DAAMMasks,
    "DAAMMaskCleanup": DAAMMaskCleanup,
    "DAAMExplorer": DAAMExplorer,
    "DAAMGuide": DAAMGuide,
    "DAAMMerge": DAAMMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAAMKSampler": "DAAM KSampler (window)",
    "DAAMSamplerCustom": "DAAM Sampler Custom (window)",
    "DAAMMasks": "DAAM Masks",
    "DAAMMaskCleanup": "DAAM Mask Cleanup",
    "DAAMExplorer": "DAAM Explorer",
    "DAAMGuide": "DAAM Guide (steps + self-attention)",
    "DAAMMerge": "DAAM Merge (multi-seed)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
