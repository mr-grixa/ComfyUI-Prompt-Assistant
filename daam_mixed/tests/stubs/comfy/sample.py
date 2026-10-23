"""Stand-in for comfy.sample.

``sample`` walks a schedule of sigmas and calls ``drive`` for every step, with
the sigma of that step, so the tests can exercise the capture window the way a
real sampler would.
"""
import torch

DRIVE = {"callable": None}
CALLS = {"sample": 0}


def fix_empty_latent_channels(model, samples, *args, **kwargs):
    return samples


def prepare_noise(samples, seed, batch_index=None):
    return torch.zeros_like(samples)


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
           latent_image, denoise=1.0, noise_mask=None, callback=None,
           disable_pbar=False, seed=0, **kwargs):
    CALLS["sample"] += 1
    sigmas = torch.linspace(14.0, 0.0, steps + 1)
    drive = DRIVE["callable"]
    if drive is not None:
        for index in range(steps):
            drive(index, sigmas)
    return latent_image
