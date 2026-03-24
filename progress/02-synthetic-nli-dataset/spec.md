# Synthetic NLI Dataset Generator Specification (v0)

This document specifies a synthetic NLI dataset generator for the bandit task defined in `progress/01-bandit-nli-spec/spec.md`.

The generator uses a hidden symbolic world to produce premise/hypothesis pairs with gold labels that are correct by construction.

## Output Dataset Schema

Each example is a JSON object with required fields:
- `id`: unique example identifier
- `premise`: rendered premise text (multiple sentences)
- `hypothesis`: rendered hypothesis text (one sentence)
- `gold_label`: one of `entailment`, `neutral`, `contradiction`
- `split`: `train` or `eval` (optional: `test`)
- `hop_difficulty`: integer in `{0,1,2}`
- `distractor_difficulty`: integer in `{1,2,3}` (number of irrelevant premise facts)
- `background_knowledge`: one of `explicit`, `implicit`

Optional but recommended metadata fields:
- `world_seed`: seed used to generate the hidden world (for reproducibility)
- `templates_version`: identifier of the rendering template set
- `ood_tag`: optional string describing the split regime (e.g., `iid`, `entity_heldout`, ...)

## Hidden World Model

The generator maintains a hidden world `W` consisting of entities and typed facts.

Important constraint:
- Gold labels must be determinable from the **premise text** plus the declared background knowledge semantics. The generator must not rely on unstated facts about specific entities (e.g., "e1 is blue") that are not present in the premise.
- Background knowledge is allowed only about predicate semantics and the type hierarchy (taxonomy). It must be either fully explicit in the premise (`background_knowledge=explicit`) or fully implicit (`background_knowledge=implicit`).

- **Entities**: synthetic names, e.g. `e0`, `e1`, ...
  - Parameter: `n_entities`

- **Facts**: the world is a set of ground-truth facts.

We support three fact families:

1) **Attribute facts** (used for 0-hop, and for contradictions)
- Example types (choose a subset in config):
  - `color(e) in Colors`
  - `location(e) in Locations`
  - `shape(e) in Shapes`
- Constraint: each attribute is **single-valued** per entity (mutual exclusivity)
  - e.g., an entity cannot be both red and blue.

2) **Taxonomy facts** (used for 1-hop and 2-hop via taxonomy)
- There is a directed acyclic graph (DAG) / tree of types.
  - Type names are **synthetic** (e.g., `t0`, `t1`, ...) to avoid reliance on pretrained world knowledge.
  - Edge meaning: `type_child -> type_parent`.
- Each entity has a base type `type0(e)` sampled from the leaf set.
- Inference rule:
  - if `type(e)=child` and `child -> parent` then `type(e)=parent`.

3) **Binary relation facts** (used for 2-hop via left_of)
- Relation: `left_of(a,b)` over entities.
- Inference rule (transitivity):
  - if `left_of(a,b)` and `left_of(b,c)` then `left_of(a,c)`.

World size and ambiguity:
- `n_entities` controls the universe size.
- Separately, the world can contain many facts, but **the premise reveals only a subset**.
  - This is crucial for neutral examples: neutral means "unknown given the premise".

## Example Construction

Each generated example is built in three stages:

1) Sample (or generate) a hidden world `W`.
2) Construct a premise as a set of **supporting facts** plus `distractor_difficulty` irrelevant facts.
3) Construct a hypothesis and assign `gold_label` by construction.

### Concrete examples

Examples show `premise`, `hypothesis`, and `gold_label` only (other fields like `split`, `hop_difficulty`, `distractor_difficulty`, `background_knowledge` must also be present in actual JSONL).

0-hop style example:

```text
premise: e3 is red. e1 is blue.
hypothesis: e3 is red.
gold_label: entailment
```

1-hop taxonomy style example (implicit background knowledge):

```text
premise: e5 is a t7.
hypothesis: e5 is a t3.
gold_label: entailment
```

(Here the dataset-level taxonomy includes the edge `t7 -> t3`. This edge is not shown in the premise when `background_knowledge=implicit`.)

1-hop taxonomy style example (explicit background knowledge):

```text
premise: t7 is a subtype of t3. e5 is a t7.
hypothesis: e5 is a t3.
gold_label: entailment
```

2-hop left_of style example (implicit background knowledge):

```text
premise: e2 is left of e7. e7 is left of e9.
hypothesis: e2 is left of e9.
gold_label: entailment
```

2-hop left_of style example (explicit background knowledge):

```text
premise: Rule: left_of is transitive. e2 is left of e7. e7 is left of e9.
hypothesis: e2 is left of e9.
gold_label: entailment
```

Example with distractors (2 supporting facts + 2 irrelevant facts):

```text
premise: e2 is left of e7. e7 is left of e9. e1 is green. e4 is a cat.
hypothesis: e2 is left of e9.
gold_label: entailment
```

### Background knowledge mode

This dataset has a third difficulty axis:

- `background_knowledge = implicit`:
  - No background knowledge statements are included in the premise.
  - The model must infer applicable semantics from training distribution.
  - Allowed implicit background knowledge is limited to:
    - predicate semantics (e.g., that `left_of` behaves like a transitive relation)
    - the type hierarchy (taxonomy edges)

- `background_knowledge = explicit`:
  - All background knowledge required to determine the label must be stated in the premise text in full.
  - This includes:
    - taxonomy edges needed for the current example
    - transitivity rule statements needed for the current example
  - No partial disclosure: if a background-knowledge family is explicit, it must be explicit for the whole dataset split/config.

### Hop difficulty buckets

We generate three hop buckets; the bucket refers to the minimum reasoning steps required to conclude entailment (for entailment examples). Neutral and contradiction examples are matched to the same bucket by controlling the hypothesis type.

- `hop_difficulty = 0`:
  - Entailment: hypothesis is a fact explicitly stated in the premise.
  - Contradiction: hypothesis conflicts with an explicitly stated single-valued attribute in the premise.
  - Neutral: hypothesis refers to an attribute/relation not stated in the premise and not inferable by the enabled rules.

- `hop_difficulty = 1` (taxonomy only):
  - Entailment: premise states `type(e)=child`, hypothesis is `type(e)=parent` where `child -> parent` is a single edge.
  - Contradiction/neutral: constructed to match the same surface form family as entailment (type statements) but be respectively inconsistent/underdetermined.

- `hop_difficulty = 2` (mixed subtype):
  - Two-hop taxonomy: premise states `type(e)=leaf`, hypothesis is `type(e)=grandparent` where `leaf -> parent -> grandparent`.
  - left_of transitivity: premise states `left_of(a,b)` and `left_of(b,c)`, hypothesis is `left_of(a,c)`.

The 2-hop bucket is allowed to mix taxonomy-2 and left_of-2 examples.

### Distractor difficulty buckets

`distractor_difficulty in {1,2,3}` is the number of irrelevant facts included in the premise in addition to the supporting facts.

- A fact is a distractor if it does not affect the truth status of the hypothesis under the declared rules.

## Label construction rules

All labels are deterministic given the premise and the declared inference rules.

- **Entailment**:
  - Hypothesis is provable from the premise using the allowed inference rules.
  - For hop buckets 1 and 2, entailment must require exactly the intended number of reasoning steps.

- **Contradiction** (no explicit negation):
  - Hypothesis contradicts the premise by violating a mutually-exclusive, single-valued attribute constraint.
  - Example: premise implies `color(e)=red` and hypothesis states `color(e)=blue`.

- **Neutral**:
  - Hypothesis is not provable from the premise and does not contradict it.
  - Neutral hypotheses must be constructed so that the premise does not determine the attribute/relation queried.

## Rendering / Templates

Facts are rendered into natural-language-like sentences using templates.

- Parameter: number of templates per fact type (attribute/type/left_of).
- Entities use synthetic names (`e0`, `e1`, ...).
- Rendering must be deterministic given a seed.

Example templates (illustrative):
- Attribute: "{e} is {color}."
- Type: "{e} is a {type}." (where `{type}` is synthetic, e.g. `t7`)
- left_of: "{a} is left of {b}."

## Splits and OOD Controls

The generator must support:

- **IID split**: random train/eval split with shared entity/type vocab.

- **OOD split modes** (configurable):
  - `entity_heldout`: entities in eval never appear in train.
  - `taxonomy_heldout`: remove a subset of taxonomy edges or nodes from train, only present in eval.
  - `relation_pattern_heldout`: hold out a subset of left_of chain patterns (implementation-defined).

Note: OOD modes that rely on implicit background knowledge (e.g., generalizing to unseen taxonomy edges) are meaningful primarily when `background_knowledge=implicit`. When `background_knowledge=explicit`, OOD should focus on structural variation that is still stated in the premise.

Each example should carry an `ood_tag` so runs can mix or separate regimes explicitly.

## Validation checks

The generator must include programmatic validation:

- Example-level:
  - Recompute label from the rendered premise facts (or from the underlying symbolic facts used to build the premise) and verify it matches `gold_label`.
  - Verify the example satisfies the intended `hop_difficulty` and `distractor_difficulty`.

- Dataset-level:
  - Check label proportions match configured targets (within tolerance).
  - Check balance across hop/distractor buckets.

## Implementation sketch using `slam-datagen`

Generation must be implemented using `slam-datagen` (https://github.com/anton-pershin/slam-datagen). The library currently provides a pattern of:
- Hydra config under `config/`
- scripts under `slam_datagen/scripts/`
- reusable components under `slam_datagen/`

Proposed additions to `slam-datagen` (following its existing structure):

1) **New dataset module under `slam_datagen/datasets/`**
- Add `slam_datagen/datasets/synthetic_nli.py` implementing:
  - a pydantic (or dataclass) sample type for one example (matching the output schema)
  - `build_synthetic_nli_dataset(cfg) -> list[Sample]`
  - `write_synthetic_nli_dataset(samples, output_file) -> Path` (JSONL)
  - internal helpers for:
    - hidden world sampling (entities + facts)
    - entailment/contradiction/neutral construction
    - hop difficulty + distractor difficulty enforcement
    - validation (per-example checks)

2) **Export from `slam_datagen/datasets/__init__.py`**
- Add the build/write functions to the package exports (mirrors `merge_quality` and `human_messages`).

3) **Script + Hydra config**
- Add `slam_datagen/scripts/generate_synthetic_nli_dataset.py` (naming consistent with existing scripts) that:
  - loads Hydra config
  - calls `build_synthetic_nli_dataset`
  - writes JSONL via `write_synthetic_nli_dataset`
  - prints a short preview + summary stats (label and difficulty bucket counts)

- Add `config/config_generate_synthetic_nli_dataset.yaml` with parameters:
  - `random_seed`
  - dataset sizes: `n_train`, `n_eval` (and optional `n_test`)
  - target label balance
  - `n_entities`
  - `background_knowledge`: `explicit|implicit`
  - hop difficulty mixture over `{0,1,2}`
  - distractor difficulty mixture over `{1,2,3}`
  - template counts per fact type
  - split mode: `iid|entity_heldout|taxonomy_heldout|relation_pattern_heldout`
  - `output_file` and `preview_samples`

4) **Tests**
- Add unit tests under `slam-datagen/tests/`:
  - label correctness for each hop bucket
  - enforcement that 1-hop taxonomy uses exactly one edge; 2-hop taxonomy uses exactly two edges
  - distractor correctness (distractors do not affect label)

This repo will consume the generator by invoking the `slam-datagen` script to produce JSONL files matching the output schema above.
