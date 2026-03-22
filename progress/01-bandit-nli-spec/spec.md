# Bandit NLI Specification

This document is the canonical specification of the first RL problem used in this project.

It defines a contextual-bandit NLI task framed as **unconstrained generation with parsing**, and it fixes the meanings of:
- what constitutes the RL context, action, and reward
- what exactly counts as a valid label and how it is parsed
- what must be stored in the replay buffer to enable controlled off-policy experiments

Decisions that are intentionally deferred are marked as **TBD**.

## Bandit Formalization

- **Context** `x`: a Natural Language Inference (NLI) example consisting of a `(premise, hypothesis)` pair.
- **Policy** `pi_theta(a | x)`: a decoder-only language model that generates a completion `a` conditioned on a prompt derived from `x`.
- **Action** `a`: the full generated completion sequence (text and/or tokens) emitted by the model for the prompt.
  - Even though reward depends only on a parsed label, **the RL action is the full completion** so that off-policy corrections (e.g., importance sampling) are well-defined.
- **Label space** `Y = {entailment, neutral, contradiction}`.
- **Parser** `parse(a) -> (y_hat, parse_success)` maps the completion `a` to either a label in `Y` or failure.
- **Reward** `r(x, a)`: deterministic scalar reward computed from the gold label `y` and the parsed label `y_hat`:
  - If `parse_success = true`: `r = 1` if `y_hat == y` else `r = 0`.
  - If `parse_success = false`: `r = 0`.

## Prompt Template

The prompt is constructed from `(premise, hypothesis)` and requests a label from `Y`.
Here is the template:

```text
Premise: {premise}
Hypothesis: {hypothesis}
Label (entailment|neutral|contradiction):
```

Notes:
- The implementation must preserve the exact label spellings shown above.
- The model is allowed to generate extra text after the label; parsing uses only the first word of the completion.
- Default generation parameters (temperature, top-p, max_new_tokens) are **TBD**, but must be logged per sample (see Replay Buffer Schema).

## Label Set and Normalization

Canonical labels (lowercase ASCII):
- `entailment`
- `neutral`
- `contradiction`

Normalization rules:
- Parsing is **case-insensitive**.
- We do **not** use aliases: only the three canonical spellings above are accepted.

## Parsing Rules (Deterministic)

Inputs:
- Completion text `a_text` (decoded from tokens) or an equivalent text view.

Outputs:
- `parsed_label`: one of the canonical labels if successful; otherwise null/None.
- `parse_success`: boolean.

Algorithm (v0):
1. Strip leading whitespace from `a_text`.
2. Extract the first token by splitting on **whitespace and punctuation**.
   - Operationally: skip any leading whitespace/punctuation, then take the next contiguous run of letters `[A-Za-z]+`.
3. Lowercase the token and match it exactly to one of: `entailment`, `neutral`, `contradiction`.
4. If it matches: `parse_success = true`, `parsed_label = matched_label`.
5. Otherwise: `parse_success = false`, `parsed_label = None`.

Required logging:
- `parse_success` must be logged for every sample.
- If parse fails, reward is 0 by definition.

## Dataset Schema (Dataset-Agnostic)

This spec is dataset-source agnostic (synthetic vs SNLI/MNLI).

Each example must provide:
- `id`: unique identifier (string or int)
- `premise`: string
- `hypothesis`: string
- `gold_label`: one of the canonical labels in `Y`
- `split`: train / eval (and optionally test)

## Replay Buffer Record Schema

Each stored record corresponds to one bandit interaction (one prompt, one completion).

Minimum required fields:
- **Identifiers / time**
  - `actor_id`: identifier of the actor process/worker that generated the sample.
  - `actor_step`: strictly contiguous per actor, incremented by 1 per interaction.
    - An interaction is uniquely identified by the pair `(actor_id, actor_step)`.
    - If an actor generates a batch of size `B` in one loop, assign steps `s, s+1, ..., s+B-1` to those `B` interactions.
  - `dataset_id`: original dataset example id (if available).
  - `learner_step_at_insert`: learner step observed when the sample was inserted (optional).

- **Task content**
  - `prompt_text`: exact prompt string used for generation.
  - `gold_label`

- **Generated action**
  - `completion_text`: full generated completion
  - `completion_token_ids`: token ids (recommended)
  - `max_new_tokens`: value used for this sample

- **Parsing + reward**
  - `parsed_label`
  - `parse_success`
  - `reward`

- **Behavior-policy bookkeeping (for off-policy analysis)**
  - `behavior_policy_id`: identifier of the actor policy snapshot (e.g., checkpoint step/hash)
  - `logp_behavior`: log-probability of the sampled completion under the behavior policy
    - definition: `logp_behavior = sum_t log pi_behavior(token_t | prompt, token_<t)`
    - store either the scalar sum (required) and optionally per-token logprobs (optional)

- **Actor sampling parameters (must be logged)**
  - `temperature`: **TBD default**, but the per-sample value must be stored
  - `top_p`: **TBD default**, but the per-sample value must be stored
  - `top_k`: optional
  - any other decoding controls used (e.g., repetition penalty)

## Off-Policy Quantities and Computation

Definitions are given for a learner update step `t` using a replayed sample collected under a behavior policy snapshot.

- **Staleness (policy lag)**
  - Let `behavior_policy_step` be the learner step (or checkpoint step) associated with `behavior_policy_id`.
  - Let `learner_step_at_update` be the current learner step when the sample is used for an update.
  - Define staleness: `Delta = learner_step_at_update - behavior_policy_step`.
  - Staleness must be computable from logged fields.

- **Importance ratio (sequence-level)**
  - Recompute `logp_target` for the stored completion under the current policy:
    - `logp_target = sum_t log pi_target(token_t | prompt, token_<t)`
  - Define importance ratio:
    - `w = exp(logp_target - logp_behavior)`
  - Idea: `w` measures how off-policy a replayed action is. Large or highly variable `w` implies high-variance gradient estimates and is a common instability mechanism in replay-based policy optimization.
  - Clipping convention: **TBD** (e.g., `w_clipped = min(w, w_max)`).

- **Effective sample size (ESS)**
  - For a batch of importance weights `{w_i}` define ESS as:
    - `ESS = (sum_i w_i)^2 / (sum_i w_i^2)`
  - Idea: ESS is a scalar summary of weight degeneracy. When ESS is low, a batch of size `N` behaves like far fewer effective samples, which typically correlates with noisy/unstable learning.
  - Report ESS and optionally normalized ESS (`ESS / batch_size`).

- **Drift metric**
  - Report a drift statistic between target and behavior policy on the sampled actions.
  - Minimum required drift: `logp_target - logp_behavior` summary statistics.
  - Idea: drift makes policy mismatch explicit. It helps explain why certain staleness/replay settings fail and provides a target for stabilizers like KL regularization.
  - Optional: KL estimates (reference choice is **TBD**).

## Required Metrics to Log/Report

Per training run (tracked over time):
- Held-out accuracy / mean reward on an eval split.
- Parse success rate.
- Staleness distribution summaries (mean/median/p95 of `Delta`).
- Importance sampling diagnostics:
  - mean/variance of `w`
  - clipped fraction (if clipping is enabled)
  - ESS (and normalized ESS)
- Stability signals:
  - NaN/Inf events
  - gradient norm / update norm (if available)
