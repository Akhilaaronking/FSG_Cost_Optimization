# FSG Cost Optimization Thesis Project

Step 1 starter repository for:

**LLM-Assisted Generative Cost–Design Co-Optimisation of the Formula Student Vehicle Bill of Materials**

## Current stage

 Parametric BOM data contract.

This starter intentionally contains **dummy engineering IDs** such as `TEST_MATERIAL_01`.
They will be replaced with  verified FSG 2026 data later.

Cost and mass are deliberately not stored as authoritative BOM inputs.
They will be computed by deterministic tools in later stages.

## Quick start

Open a terminal in this folder.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py main.py
py -m pytest -q
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
python -m pytest -q
```

## Expected validation output

`python main.py` should finish with:

```text
BOM VALIDATION: PASS
```

`pytest -q` should show all tests passing.

## Do not do yet

Do not add:
- LLMs
- RAG
- LangGraph
- NSGA-II
- QLoRA
- supplier APIs

We will add them only after the data contract is stable.
