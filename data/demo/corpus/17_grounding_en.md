# Grounding and Citation Checks

## Numbered Sources
Prompts should present context as `[1]`, `[2]`, … and require the model to cite those indices. If evidence is missing, the model must refuse rather than invent.

## Post-Generation Checks
Verify every `[n]` exists. Stronger checks confirm that numbers and named entities in the answer appear in the cited chunk. Mark violations in the UI and in metrics.

## Unanswerable Questions
At least part of an evaluation set should be unanswerable given the corpus. A correct system answers with an explicit “not in the knowledge base” style refusal.
