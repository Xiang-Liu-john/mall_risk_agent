# Evaluation Harness

This folder contains deterministic evaluation utilities for the Agentic BI prototype.

Run:

```bash
python -m evaluation.run_evaluation
```

The runner writes `evaluation/latest_results.json` only from actual local execution. Do not copy numbers into project docs unless the command completed in the current environment.

Metrics:

- Agent: Intent Accuracy, Filter Accuracy, Tool Selection Accuracy, Task Success Rate.
- Retrieval: Hit@K, Recall@K, MRR against expected evidence source categories.
- Generation: required section compliance, citation presence, approval boundary compliance, store fact coverage.

LLM-as-a-Judge is intentionally not enabled in this prototype. If added later, keep judge prompts and outputs in a separate module and report judge scores separately from deterministic metrics.
