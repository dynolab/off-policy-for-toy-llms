# Synthetic NLI Dataset

This task designs and implements a synthetic dataset generator for the contextual-bandit NLI problem specified in `progress/01-bandit-nli-spec/spec.md`.

## Goal

Create a dataset generator that produces `(premise, hypothesis, gold_label)` triples with labels that are correct by construction, plus explicit difficulty controls that we can later use for off-policy experiments.

We use a hidden symbolic world to generate facts, render a subset of them as the premise, and then construct hypotheses that are entailments, contradictions, or neutral w.r.t. the premise.

## Deliverables

- `progress/02-synthetic-nli-dataset/spec.md`: generator specification (world model, rules, difficulty knobs, split strategy, validation checks).
- Code that implements the generator and writes datasets in a stable format (path/TBD; should live under `code/` and/or inside this task directory).
- A small generated dataset artifact for smoke-testing (size/TBD) and basic summary statistics.

## Expectations for `spec.md`

`spec.md` must define, at minimum:

- **Output dataset schema**
  - Required fields: `id`, `premise`, `hypothesis`, `gold_label`, `split`, `hop_difficulty`, `distractor_difficulty`, `background_knowledge`.
  - Labels must match the canonical label strings from Task 01 (`entailment`, `neutral`, `contradiction`).

- **Hidden world model**
  - Entity set with synthetic names.
  - Fact types to support:
    - 0-hop attribute facts (e.g., `color(x)=red`, `location(x)=kitchen`, etc.).
    - Taxonomy facts (for 1-hop entailment): `type(x)=leaf` with a taxonomy graph (leaf -> parent).
    - Binary relation facts (for 2-hop entailment): `left_of(a,b)` with transitive closure used for entailment.

- **Difficulty knobs (must be explicit parameters)**
  - Dataset size: `n_train`, `n_eval` (and optional `n_test`).
  - Label balance: explicit target proportions (default: 1/3 each).
  - Hop difficulty: exactly three buckets: `0-hop`, `1-hop`, `2-hop`.
    - 1-hop: entailment requires exactly one taxonomy edge.
    - 2-hop: entailment requires exactly two steps (e.g., taxonomy 2 edges or left_of transitivity).
  - Distractor difficulty: exactly three buckets by number of irrelevant premise facts: `1`, `2`, `3`.
  - Background knowledge mode: `explicit` vs `implicit`.
    - Explicit: all background knowledge is stated in the premise text.
    - Implicit: no background knowledge is stated in the premise text.
  - World size: `n_entities`.
  - Template variation: number of rendering templates per fact type.

- **Construction rules for labels (correct-by-construction)**
  - Entailment: hypothesis is implied by premise using the enabled rules and the chosen hop bucket.
  - Contradiction: hypothesis contradicts premise via mutually-exclusive attributes (no explicit negation).
  - Neutral: hypothesis is neither entailed nor contradicted given the premise and enabled rules.

- **Splits and OOD controls**
  - IID split: standard random train/eval split.
  - OOD splits as explicit options/parameters (at least):
    - entity-heldout (eval entities disjoint from train)
    - taxonomy-edge/node-heldout (parts of the taxonomy withheld)
    - relation-pattern-heldout for `left_of` chains (as applicable)

- **Validation checks**
  - Unit-testable checks that each generated example satisfies its gold label under the declared semantics.
  - Dataset-level checks: label balance, per-bucket balance, and that hop/distractor metadata matches the requested bucket.

## Notes / deferred details

- Exact set of attributes (e.g., colors, locations) and taxonomy graph size are TBD but must be specified in `spec.md` before main experiments.
- We postpone explicit artifact/confound injection controls to a later task.
