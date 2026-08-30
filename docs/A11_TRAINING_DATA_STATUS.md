# A11 Training Data Status

A11 prepares C3 training-data files only. It does not train a model, install MLX, run C3, or produce C3 experimental results.

## Source

- `data/benchmark/real_search_space.json`
- `data/benchmark/pilot_10_parts_ground_truth.json`
- `schemas/proposal.schema.json`
- `schemas/ollama_proposal_output.schema.json`

The dataset is programmatically generated instruction data derived from an engineering-reviewed admissible search space. It is not a hand-written engineering judgement dataset.

## Methodology

Positive examples use only explicit, non-baseline material/process alternatives approved for the target part in the real search space. Negative examples teach rejection of inadmissible identifiers, unknown identifiers, target-field mistakes, object-valued material/process `new_value` payloads, PA6 sheet/filament conflation, and recorded turning/non-axisymmetric conflicts.

Assistant positives make no numerical cost, mass, strength, or Formula Student compliance claims. They only select an engineering-reviewed candidate for later deterministic evaluation.

## Split

Grouped split by `part_id`, deterministic with split seed `11`:

- Train: PILOT_001, PILOT_002, PILOT_003, PILOT_004, PILOT_005, PILOT_006, PILOT_007, PILOT_008
- Validation: PILOT_009
- Test: PILOT_010

No part ID appears in more than one split.

## Counts

- Total examples: 1512
- Train examples: 1212
- Validation examples: 138
- Test examples: 162
- Positive examples: 468
- Negative examples: 1044
- Material examples: 900
- Process examples: 612

## Hashes

- Source hash: `5c936852d2fac826edc2c8a7a3e40c03d2f23dd9192352b332a162cdf1ec060a`
- Converted search-space hash: `f3ff489076fe7f49ca6e6c5444e23a3af7551997b2ac4236a67d71390ee89c68`
- Benchmark hash: `0a3ffac413e82ddd7d5c6ae39b25e33267b5a930034b4c94cc74d05d89cc5146`
- Dataset hash: `e7870a5729ca1dd6ce7e713a846e48250a2967be6a2ee651d98a1780ff4169ce`

## Limitations

- Generated examples are not independently verified optimal designs.
- They do not establish global optimality, final cost/mass performance, or H2 conclusions.
- Only material and process changes are represented.
- Interpretive Formula Student rule compliance is not encoded as a deterministic training target.
