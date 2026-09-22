"""Per-cell schedule, source and run-identity validation for GH125."""

import hashlib
import struct
from grok1_alpha_schedule_contract import (
    BLOCKS,
    CELLS,
    PROTOCOL,
    SEED,
    TOKENS,
    TOKEN_IDS_SHA256,
    TOP_K,
    integer,
    number,
    require,
    sha,
)
from grok1_alpha_schedule_measurements import _finite_evidence, _metrics, validate_resources


def _validate_settings(chain, spec):
    settings = {
        "tokens": TOKENS,
        "token_seed": SEED,
        "top_k": TOP_K,
        "blocks": list(BLOCKS),
        "skip_fp16_control": False,
        "hp_blocks": list(spec.hp),
        "int4_blocks": list(spec.quantized),
        "channel_alpha_blocks": list(spec.channel_alpha),
        "expert_mode": spec.mode,
        "ternary_blocks": [],
        "hp_period": None,
        "hp_schedule_kind": "explicit",
    }
    for key, value in settings.items():
        require(
            chain.get(key) == value and type(chain.get(key)) is type(value),
            f"unexpected per-cell setting: {key}",
        )
        if isinstance(value, list):
            require(all(integer(v) for v in chain[key]), f"noninteger block list: {key}")


def _validate_chain_metrics(chain):
    require([r["block"] for r in chain["per_block"]] == list(BLOCKS), "invalid block order")
    for row in chain["per_block"]:
        require(integer(row["block"]), "noninteger metric block identifier")
        _metrics(row["expert_only"])
        _metrics(row["fp16_control"], control=True)
    for key in ("expert_only_chain_exit", "fp16_chain_exit"):
        end = chain["end_of_chain"][key]
        require(number(end.get("residual_drift_relative_norm"), 0), "missing terminal drift")
        require(number(end.get("residual_cosine"), -1, 1 + 1e-9), "invalid terminal cosine")


def _validate_scale_map(mapping, source, block, detail):
    experts = {k: v for k, v in mapping.items() if "moe_expert." in k}
    require(len(experts) == len(mapping) == 3, detail)
    require({k.rsplit(".", 1)[-1] for k in experts} == {"gate", "down", "up"}, detail)
    require(all(k.startswith(f"block_{block:03d}.") for k in experts), detail)
    require(set(experts.values()) == {source}, detail)


def _validate_pack(row, block, spec, cell):
    require(sha(row["pack_sha256"]) and sha(row["npy_sha256"]), "invalid input hash")
    require(
        row["container_versions"] == [3] and all(integer(v) for v in row["container_versions"]),
        "pack_v2 scales require the canonical v3 container",
    )
    require(
        set(row["scale_sources"]) == set(row["pack_scale_sources"]),
        "applied and pack scale tensor identities differ",
    )
    for field, source in (
        ("scale_sources", spec.scale_source(block)),
        ("pack_scale_sources", "pack_v2"),
    ):
        _validate_scale_map(row[field], source, block, f"invalid {field} for {cell}/{block}")


def _validate_packs(packs, spec, cell):
    require([r["block"] for r in packs] == list(BLOCKS), "invalid source coverage")
    for block, row in enumerate(packs):
        require(integer(row["block"]), "noninteger source block identifier")
        _validate_pack(row, block, spec, cell)


def _validate_token_ids(provenance):
    ids = provenance["token_ids"]
    require(isinstance(ids, list) and len(ids) == TOKENS, "invalid token IDs")
    require(all(integer(i) and 0 <= i < 131072 for i in ids), "invalid token IDs")
    require(
        hashlib.sha256(struct.pack(f"<{TOKENS}q", *ids)).hexdigest()
        == provenance["token_ids_sha256"]
        == TOKEN_IDS_SHA256,
        "token IDs/order differ from frozen sample",
    )


def _validate_provenance(provenance):
    for field in ("embedding_sha256", "token_ids_sha256", "checkpoint_identity"):
        require(sha(provenance.get(field)), f"invalid {field}")
    _validate_token_ids(provenance)
    impl = provenance["implementation"]
    require(
        sha(impl.get("commit"), 40) and impl.get("dirty") is False,
        "implementation must be a clean commit",
    )
    require(
        isinstance(provenance["runtime"], dict) and bool(provenance["runtime"]), "missing runtime"
    )


def validate_cell(payload, cell, run_id):
    """Reject malformed evidence before any same-run comparison."""
    try:
        spec = CELLS[cell]
        _finite_evidence(payload, cell)
        require(
            payload["protocol"] == PROTOCOL and payload["cell"] == cell, "wrong protocol or cell"
        )
        require(
            isinstance(run_id, str) and bool(run_id) and payload["run_id"] == run_id,
            "stale run identity",
        )
        chain = payload["chain"]
        _validate_settings(chain, spec)
        _validate_chain_metrics(chain)
        _validate_packs(chain["pack_provenance"], spec, cell)
        _validate_provenance(payload["provenance"])
        validate_resources(payload["resources"])
        fp = payload["resources"]["fp16_expert_payload_bytes"]
        require((fp > 0) == bool(spec.hp), "resource payload contradicts HP schedule")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"malformed {cell} evidence: {exc}") from exc
    return payload
