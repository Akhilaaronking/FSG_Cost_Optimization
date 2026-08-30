# A8 Real Search-Space Status

The A8 weighted-sum and NSGA-II implementations are operational.

An engineering-reviewed Formula Student material/process search space is now present at `data/benchmark/real_search_space.json`.

The real search-space adapter in `src/optimization/search_space.py` converts the approved Option A fields `admissible_materials` and `admissible_processes` into optimiser-facing `material_choices` and `process_choices`. It cross-checks the approved current material/process against the frozen B4 benchmark and validates every active choice against the canonical registries.

Membership in `materials.csv`, `processes.csv`, or `fasteners.csv` proves identifier existence only; it does not prove that a material, process, or fastener is engineering-interchangeable for a given Formula Student part.

Current active real optimisation scope:

- material choices
- process choices

Not active yet:

- geometry variables, unless explicit numeric engineering-verified bounds are added
- fastener variables

Synthetic validation must still not be reported as C5 Formula Student experimental results.
