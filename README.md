# off-policy-for-toy-llms

Code & notes on the Off-Policy for Toy LLMs project

## Project description

**Topic**: Off-Policy RL for Toy LLMs

**Description**: This project studies off-policy effects in asynchronous reinforcement learning for small decoder-only LLMs under tight compute constraints. We focus on a replay-buffer setup where fresh trajectories are continuously collected by an actor and appended to a buffer while a learner updates the policy from a mixture of fresh and replayed data. The initial task family is contextual bandits framed as unconstrained generation with parsing, starting from NLI label prediction (entailment / neutral / contradiction). For v0, we require that the first token of the generated completion (case-insensitive; tokenization splits on whitespace and punctuation) is the label. We run controlled experiments that vary off-policiness (policy lag, replay intensity, behavior-policy mismatch) and compare objectives such as naive replay-based updates vs importance sampling (with clipping) and KL-regularized variants, measuring stability, final accuracy, and compute cost.

## Research objective

### Abstract

We will build a research-friendly asynchronous RL framework with a replay buffer for small LLMs and use it to quantify how off-policiness impacts learning. Using contextual bandit NLI tasks with deterministic rewards derived from parsing model outputs, we will run controlled sweeps over policy staleness and replay intensity and evaluate mitigation methods (importance sampling with clipping, KL regularization). We will log off-policy diagnostics (staleness, drift/KL, IS weight statistics, effective sample size) to connect failure modes to measurable causes.

### Key research question

How do controllable sources of off-policiness in an async replay-buffer RL pipeline (policy lag, replay intensity, behavior sampling mismatch) affect training stability, sample/compute efficiency, and final accuracy for small LLMs, and which corrections/stabilizers (IS-clipping, KL regularization) most reliably prevent degradation?

### Why this deserves studying

Replay-buffer and asynchronous data collection are common in practical RL systems, but off-policy effects are often confounded by long-horizon credit assignment and noisy rewards. Contextual bandits with deterministic evaluation provide a clean, cheap testbed where off-policiness can be precisely controlled and measured, enabling clear conclusions about stability regimes, failure modes, and the cost/benefit of off-policy corrections for small LLM training.
