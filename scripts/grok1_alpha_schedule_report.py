"""Human-readable rendering of validated GH125 outcome records."""

from grok1_alpha_schedule_contract import CELLS, CONTRAST_KEYS


def _contrast_row(name, row):
    vals = [row["values"][cell] for cell in CELLS]
    formatted_vals = ["empty band" if v is None else f"{v:.9g}" for v in vals]
    contrasts = row.get("contrasts")
    if contrasts:
        formatted_deltas = [f"{contrasts[key]:.9g}" for key in CONTRAST_KEYS]
    else:
        formatted_deltas = ["unavailable"] * len(CONTRAST_KEYS)
    return (
        "| "
        + " | ".join([name, row["favorable_direction"], *formatted_vals, *formatted_deltas])
        + " |"
    )


def _attribution_lines(contrasts):
    lines = []
    for name, row in contrasts.items():
        if row.get("observed_effects"):
            lines.append(
                f"- {name}: "
                + "; ".join(f"{k}: {row['observed_effects'][k]}" for k in CONTRAST_KEYS)
            )
    return lines


def render_report(payload):
    lines = [
        "# Grok-1 alpha/schedule ablation",
        "",
        f"Status: {payload['status']}",
        f"Run: {payload['run_id']}",
        "",
        "The authoritative record is outcome.json.",
        "",
    ]
    if payload["status"] != "complete":
        return "\n".join(
            lines
            + [
                "No causal conclusion: incomplete evidence.",
                str(payload.get("error", "running")),
                "",
            ]
        )
    lines += [
        payload["interpretation"],
        "",
        payload["precision_cost_note"],
        "",
        payload["control_semantics"],
        "",
        "Top-1 0.95 remains diagnostic only.",
        "",
        "| Metric | Favorable | A | B | C | D | B-A | D-C | C-A | D-B | Interaction |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in payload["paired_contrasts"].items():
        lines.append(_contrast_row(name, row))
    lines += [
        "",
        "Observed attribution by metric (signs describe this run, not statistical significance):",
        "",
    ]
    lines.extend(_attribution_lines(payload["paired_contrasts"]))
    lines += [
        "",
        "Raw per-block controls, source identities and resource accounting are in outcome.json.",
        "",
    ]
    return "\n".join(lines)
