"""QED-RF — autonomous adaptive-robustness audit of AI-native beam prediction.

Transfers QED's principle ("report a worst case you can recompute, not a claim")
to adversarial ML for 6G physical-layer AI: a robustness claim is accepted only
if it survives the *strongest realizable* attack the audit agent can find, as
measured by the physical spectral-efficiency oracle. It quantifies the gap
between the robustness a defense *reports* (under a single fixed-budget attack)
and the robustness it *actually* has (under an escalating adaptive audit with a
transfer control) — the "false sense of security" lesson (Athalye et al.)
applied to AI-native beam management. Every reported violation carries a
recomputable per-user witness tuple.

Substrate (models, channels, attacks, SE oracle) is reused read-only from the
lab's `usama/paper2026` codebase via `rfcore`.
"""

__version__ = "0.2.0"
