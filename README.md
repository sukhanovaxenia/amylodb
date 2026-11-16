# amylodb

A unified amyloid database that **reconciles two independently developed data
lines** into one versioned, ML-ready dataset:

1. **Curated sources** — nine curated amyloid databases (WALTZ-DB, Cross-Beta DB,
   AmyLoad, AmyloidExplorer, AmyloidAtlas, Amylobase, AmyPro, AmyloGraph,
   CPAD peptides + structures), parsed into a common record with an 11-level
   structural classification and physicochemical features.
2. **Literature mining** — protein-level extractions mined from PubMed abstracts.
   The recommended `models` mode combines OpenBioNER-v2 (typed NER), the PDBEurope
   protein-structure NER (experimental methods), mDeBERTa-v3 zero-shot NLI
   (aggregation, reversibility, structure, functional class, LLPS) and an
   optional LLM only for fields NLI leaves undecided. All models are frozen and
   run on CPU or Apple GPU.

The two lines use different units (curated records are **peptide/region**
centred; mined records are **protein** centred), so amylodb does not merge them
mechanically. It lifts both into one schema, keeps their identity notions
apart, and reconciles overlaps with tier-weighted voting. Every record carries
its **provenance**, a single ordered **evidence tier**, and a **prion class**.

Outputs feed two downstream projects: the amyloidogenicity predictor
(`amylobert`) consumes the fixed feature matrix, and the graded
classification graph consumes the graph export.

## Architecture

```
curated sources ──▶ parsers ─┐
                             ├─▶ lift to AmyloidRecord ─▶ reconcile ─▶ features (444-wide)
literature ─▶ mining ─▶ adapter ┘        (level-aware)   (tier vote)  ├─▶ graph (nodes + edges)
(PubMed/NER/NLI/LLM, on host)                                          └─▶ records.tsv / sqlite / fasta
```

- **Curated path** runs anywhere, no network or models required.
- **Mining path** runs on a host with PubMed / UniProt / model access and
  writes a JSON of extractions, which is fed back in via `--mined`.

## Install

```bash
pip install -e ".[ml]"          # core + downstream ML baselines
pip install -e ".[mining]"      # add PubMed/NER/NLI/LLM mining (host with models)
pip install -e ".[dev]"         # tests + lint
```

## Quickstart

```bash
# 1. edit configs/default.yaml with your curated source paths
# 2. (optional) run mining on a host with model access -> mined.json
# 3. build
python -m amylodb build --config configs/default.yaml
# or explicitly:
python -m amylodb build --waltzdb waltzdb.csv --amyload AmyLoad.csv --out build/out
```

### Outputs (under `output_dir`)

| File | Content |
|------|---------|
| `records.tsv` | one row per consensus record: metadata, provenance, record level, prion class, evidence tier, mined fields |
| `features.tsv.gz` | fixed **444-wide** feature matrix keyed by `record_id` (21 physicochemical + 20 AA + 400 dipeptide + 3 grouped) |
| `graph.json` | nodes (records + protein hubs) and edges (`belongs_to`, `interaction`, `co_mention`) for the graded-classification graph |
| `conflicts.json` | region groups whose label conflict was too close to resolve |
| `reconcile_stats.json` | counts: input, groups, consensus, conflicts, levels, prion-like |

## The model: provenance, evidence tiers, prion class

- **Provenance** — `curated`, `mined`, or `predicted`.
- **Evidence tier** (ordered, higher = stronger): `COMPUTATIONAL < TEXT_MINED <
  LITERATURE_CURATED < STAINING_BINDING < KINETIC < STRUCTURAL`. A literature-mined
  claim never outranks a structurally resolved one; the tier is the single
  source of truth for the consensus voting weight.
- **Prion class** — `not_prion`, `prion_like_domain`, `transmissible_prion`.
  Prion-like-domain RBPs (FUS, hnRNPA1, TDP-43, PRRC2A/B) are **kept and tagged**,
  never dropped, because they are central to translation-coupled RNP condensates.
  Only transmissible mammalian prion pathology can be excluded, and only on
  request (`--exclude-transmissible-prion`).

### Candidate eligibility and evidence of self-aggregation

The curator's decision of 2026-11-14 distinguishes an **amyloidogenicity
candidate** from a protein with demonstrated self-aggregation. At least one
source-specific observation of the protein or a defined region may support
candidate eligibility: aggregation, condensate formation,
a low-complexity region (LCR), an aggregation-prone region (APR), or a
computational prediction of aggregation. Record the entity, source and method
behind each basis. A condensate, LCR/APR or prediction alone does not establish
amyloid formation.

Review candidate eligibility separately from evidence of the protein's own
amyloid aggregation. An appropriately marked `model_only` record with a positive
model vote may be retained as a candidate. Text classification by NLI is distinct
from sequence-based prediction; neither should be presented as experimental
confirmation. "Not established in the reviewed material" is not a negative
experimental result. Entity, organism, role and subtype errors remain errors;
co-mention or regulation of another protein's assembly does not establish the
candidate's own property.

This decision does not select new LCR/APR methods or thresholds, change functional
roles, or modify the saved database automatically. The amylodb-004 audit compares
all **1870 v3e rows against their saved status/evidence**, not historical y/n
correctness marks. It identifies **15 text-supported inclusion proposals**
(13 aggregation, 2 LCR), **9 conditional NLI-only proposals**, and **39 unresolved
rejected rows**. The other 1807 retain their status in this bounded audit; this
is not a new certification of their scientific correctness. No database changes
were applied. See [`docs/changes_from_source.md`](docs/changes_from_source.md),
section 9.15, for provenance, the NLI false-positive control, and limitations.
Candidates without confirmed amyloid evidence remain excluded from training
sets that require confirmed positives.

See `docs/running.md` for setup, model download (`scripts/fetch_models.py`),
model validation (`scripts/check_models.py`), mining and the quality audit
(`scripts/audit_mining.py`); `docs/mining.md` for how claims become fields; and
`docs/changes_from_source.md` for the differences from the two source branches
and the measured defects behind them.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## License

MIT © 2026 Xenia Sukhanova. ORCID 0000-0001-7022-0832.
