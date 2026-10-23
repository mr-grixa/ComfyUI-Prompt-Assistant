"""Stand-in for comfy.samplers: only the name lists the node schema needs."""


class KSampler:
    SAMPLERS = ["euler", "dpmpp_2m"]
    SCHEDULERS = ["normal", "karras"]
