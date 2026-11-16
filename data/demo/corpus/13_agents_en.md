# Multi-Agent Research Assistant Patterns

## Roles
An orchestrator plans and synthesizes. A retriever agent searches the knowledge base. A web researcher fetches external pages under policy. An analyst compares facts. A critic checks citations and completeness.

## Tool Policy Gate
Every side-effecting tool call passes a policy gate. Modes include auto, on_request, ask, and handoff. Unknown tools fail closed to ask. Approvals are audited.

## Run State Machine
Runs move through queued, planning, running, awaiting_approval, awaiting_subagents, then completed, failed, cancelled, or timed_out.

## Budgets
Limit steps, tokens, cost, and active seconds. On exhaustion, force a partial synthesis instead of crashing silently.
