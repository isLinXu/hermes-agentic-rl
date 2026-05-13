"""Back-compat shim.

The symbol `AtroposGrpoTrainer` was misleadingly named — it never performed any
gradient update. The real implementation now lives in
`hermes_agentic_rl.exporters.atropos_jsonl` as `AtroposJsonlExporter`.

Importing from here still works (with a DeprecationWarning) for existing users
and tests. New code should import from `exporters.`.
"""

from hermes_agentic_rl.exporters.atropos_jsonl import (
    AtroposGrpoTrainer,
    AtroposJsonlExporter,
)

__all__ = ["AtroposGrpoTrainer", "AtroposJsonlExporter"]
