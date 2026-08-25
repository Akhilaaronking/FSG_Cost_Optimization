# A8 Real Search-Space Status

The A8 weighted-sum and NSGA-II implementations are operational and are tested with synthetic algorithm fixtures only.

No engineering-verified Formula Student optimisation search space was found in the repository. The repository contains a changelog reference to `data/benchmark/development_30_parts.json`, but that file is not present.

Real C5 optimisation is therefore blocked until an explicit admissible design/search-space definition is supplied. Membership in `materials.csv`, `processes.csv`, or `fasteners.csv` proves identifier existence only; it does not prove that a material, process, or fastener is engineering-interchangeable for a given Formula Student part.

Person B must supply, at minimum:

- baseline part IDs covered by the real development benchmark
- allowed material alternatives per part
- allowed manufacturing-process alternatives per part
- any material/process compatibility restrictions per part
- geometric decision variables and verified lower/upper bounds
- fastener alternatives and quantity bounds where fasteners are mutable
- whether each search-space entry is engineering verified
- source/provenance notes for the admissibility decisions

Until those inputs exist, synthetic validation must not be reported as C5 Formula Student experimental results.
