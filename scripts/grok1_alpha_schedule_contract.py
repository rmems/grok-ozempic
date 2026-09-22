"""Frozen GH125 four-cell contract. Historical GH85 decisions are not inputs."""

from dataclasses import dataclass
import math
import re
from types import MappingProxyType

PROTOCOL = "grok1-int4-alpha-schedule-v1"
BLOCKS = (0, 1, 2, 3)
TOKENS = 8192
SEED = 20260806
# Public checksum of the ordered sample; not an authentication secret.
TOKEN_IDS_SHA256 = "57b0e364bb25ac4bd9047b592e28ae1470af94481be073a1f963c7cd0a5ee3de"  # nosec B105
TOP_K = 2
MARGIN_BANDS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.5), (0.5, 1.01))
RESOURCE_SCOPE = "measured-block expert payload; not whole-model compression"
INTERACTION = "(D-C)-(B-A)"
CONTRAST_KEYS = ("B-A", "D-C", "C-A", "D-B", INTERACTION)


@dataclass(frozen=True)
class Cell:
    mode: str
    hp: tuple[int, ...]

    @property
    def quantized(self):
        return tuple(b for b in BLOCKS if b not in self.hp)

    @property
    def channel_alpha(self):
        return self.quantized if self.mode == "int4_channel_alpha" else ()

    def scale_source(self, block):
        if block in self.hp:
            return "fp16_control"
        return "research_int4_channel_alpha_side" if self.channel_alpha else "research_int4_side"


CELLS = MappingProxyType(
    {
        "A": Cell("int4", ()),
        "B": Cell("int4_channel_alpha", ()),
        "C": Cell("int4", (1, 2, 3)),
        "D": Cell("int4_channel_alpha", (1, 2, 3)),
    }
)
DIRECTIONS = MappingProxyType(
    {
        "block_output_cosine": "higher",
        "moe_output_cosine": "higher",
        "router_top1_agreement": "higher",
        "router_top2_set_agreement": "higher",
        "incoming_residual_drift": "lower",
        "chain_exit_drift": "lower",
        "chain_exit_cosine": "higher",
        "top1_flip_rate": "lower",
    }
)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def integer(value):
    """Integer evidence excludes booleans and integral floats."""
    return isinstance(value, int) and not isinstance(value, bool)


def number(value, lo=-math.inf, hi=math.inf):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and lo <= value <= hi
    )


def sha(value, size=64):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value) is not None
