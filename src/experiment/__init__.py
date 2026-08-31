"""
A12 unified experiment harness.

Runs conditions C1, C2, C3 and C5 under one run-identity model
(thesis 11.4), one event log (11.16), one metrics contract, and one
equal-budget definition (11.6). See docs/A12_EXPERIMENT_HARNESS_DESIGN.md.

Build order (docs/A12 section 11):
    1. identity.py   -- this step: run identity + run_config.json
    2. ledger.py
    3. apply_proposal.py
    4. events.py
    5. probe.py
    6. drivers.py
    7. metrics.py
"""
