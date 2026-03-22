# Bandit NLI Spec

This task freezes the definition of the first training problem we will use to study off-policy effects for toy LLMs.

## Goal

Produce a single canonical specification of a contextual-bandit NLI task framed as unconstrained generation with parsing, such that later implementation choices cannot silently change what the “RL action”, reward, or off-policy measurements mean.

## Deliverables

- `progress/01-bandit-nli-spec/spec.md`: canonical problem specification.

## Expectations for `spec.md`

`spec.md` must define, at minimum:

- **Bandit formalization**
  - Context `x` (premise/hypothesis prompt)
  - Action `a` as the full generated completion sequence (not just the parsed label)
  - Parsed label `y_hat = parse(a)`
  - Reward `r` as a deterministic function of `y_hat` and the gold label

- **Prompt template**
  - Exact prompt formatting and the canonical label strings to be produced.

- **Label set and normalization**
  - Canonical labels: `entailment`, `neutral`, `contradiction`
  - Any accepted aliases and how they map to canonical labels (or explicitly state that none are allowed).

- **Parsing rules (deterministic)**
  - Text normalization (e.g., lowercasing)
  - Rule for extracting the label if multiple appear
  - Behavior on parse failure (must be fully specified)
  - Outputs: `parsed_label` and `parse_success` flag

- **Replay buffer record schema** (fields required for off-policy analysis)
  - Prompt, completion, gold label, parsed label, reward, parse_success
  - `logp_behavior(completion | prompt)` (sequence-level; per-token optional)
  - Behavior policy identifier (e.g., checkpoint step/hash)
  - Actor sampling parameters used to generate the completion (even if defaults are TBD)
  - Timestamps/steps sufficient to compute staleness

- **Off-policy quantities and how to compute them**
  - Staleness definition (policy version lag)
  - Importance ratio definition `w = exp(logp_target - logp_behavior)` and any clipping convention placeholder
  - Effective sample size (ESS) definition to be reported
  - Drift metric to report (e.g., KL to a reference policy), with the reference policy options stated

- **Required metrics to log/report**
  - Held-out accuracy (mean reward)
  - Parse success rate
  - IS weight statistics (mean/var, clipped fraction, ESS)
  - Staleness distribution summaries

## Deferred decisions

The following may be left as TBD in `spec.md`, but must be called out explicitly as TBD:

- Dataset source (synthetic vs SNLI/MNLI)
- Default actor decoding parameters (temperature/top-p/max_new_tokens)
- Choice of KL reference policy (if KL-regularized objectives are used)
