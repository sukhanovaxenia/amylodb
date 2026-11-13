# Running amylodb: setup, models, build

Commands assume macOS (your MacBook) with conda/mamba available. Run them from
the repository root (`Dissertation/amylodb`).

## 1. Environment

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Documents_icloud/Lab_amyloids/Projects/Dissertation/amylodb

# create the environment (mamba is faster; conda works too)
mamba env create -f env/environment.yaml
conda activate amylodb

# install the package itself (editable), with the ML extras for downstream use
pip install -e ".[ml]"
```

Verify:

```bash
amylodb --help
python -m amylodb build --help
pytest -q            # 27 tests, no network needed
```

## 2. Build the curated database (no models, no network)

This path is fully working now. Point the config at your curated source files,
then build.

```bash
# stage the curated sources this repo already parses (from the Amyloid-DB repo)
ADB=~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Documents_icloud/Lab_amyloids/Projects/Anna_Julia/Amyloid-DB
mkdir -p data/curated/CPAD
ln -sf "$ADB"/v1/Database/waltzdb.csv                      data/curated/waltzdb.csv
ln -sf "$ADB"/v1/Database/crossbetadb.json                 data/curated/crossbetadb.json
ln -sf "$ADB"/v1/Database/AmyLoad_session_unlogged_unlogged_.csv data/curated/AmyLoad.csv
ln -sf "$ADB"/v1/dbs/AmyloidAtlas.tsv                      data/curated/AmyloidAtlas.tsv
ln -sf "$ADB"/v1/dbs/AmyloidExplorer.tsv                   data/curated/AmyloidExplorer.tsv
ln -sf "$ADB"/v1/dbs/amypro.txt                            data/curated/amypro.txt
ln -sf "$ADB"/v1/dbs/amylograph.txt                        data/curated/amylograph.txt
ln -sf "$ADB"/v1/dbs/amylobase.txt                         data/curated/amylobase.txt
ln -sf "$ADB"/v1/Database/CPAD/aggregating\ peptides.xlsx  "data/curated/CPAD/aggregating_peptides.xlsx"
ln -sf "$ADB"/v1/Database/CPAD/amyloid\ structure.xlsx     "data/curated/CPAD/amyloid_structure.xlsx"

# build
amylodb build --config configs/default.yaml
```

Outputs land in `build/amylodb_out/` (see README for the file list). On the
bundled sources this reproduces ~6,100 consensus records, a 444-wide feature
matrix and a region→protein graph.

## 3. Models for literature mining

Install the mining extras (transformers, torch with MPS on Apple Silicon,
zshot, sentencepiece, huggingface_hub) and the scispaCy model used by the
`hybrid` mode:

```bash
pip install -e ".[mining]"   # includes zshot + gliner (gliner is imported by zshot)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 3a. Download and pin the models (once)

```bash
python scripts/fetch_models.py --models-dir models
```

This downloads the four models (NLI, structure NER, triage, OpenBioNER) plus
the tokenizer zshot needs, and writes `models/models.lock.json` with the exact
revision of each. Later runs reuse these revisions; `--update` moves to the
latest. After this step mining can run offline (`export HF_HUB_OFFLINE=1`).

### 3b. Check the models before mining

```bash
python scripts/check_models.py --models-dir models
```

Three short abstracts written for this check, with answers fixed in advance:
a positive control (alpha-synuclein: amyloid, cross-beta, irreversible,
pathological, cryo-EM), a negative control (hexokinase metabolism: no record)
and a condensate case (FUS: LLPS, conditionally reversible, prion-like domain
kept). It prints PASS/FAIL per expectation with the claim scores behind each
decision and the time per abstract, and writes `models/check_report.json`.
A FAIL here is information about the models, not a crash: look at the scores
before mining.

### 3c. Access to services

| Service | How | Needed for |
|---------|-----|------------|
| PubMed (NCBI) | `--email you@example.com`; optional `--pubmed-key <key>` | always |
| UniProt REST | nothing to configure (public) | name resolution |
| HuggingFace | only for `fetch_models.py`; `huggingface-cli login` not required (models are public) | model download |
| Ollama (local LLM) | `ollama serve &` and `ollama pull qwen2.5:7b` | LLM fallback for undecided fields |
| Anthropic / OpenAI | `--llm-provider anthropic --llm-key ...` | only instead of Ollama |

No GPU is required; on the MacBook `--device auto` uses the Apple GPU (MPS).

## 4. Run mining, then feed it back

```bash
# 1. mine PubMed with the model-based extractor
python -m amylodb.mining.pipeline \
    --email you@example.com \
    --mode models --ner-model openbioner --models-dir models --device auto \
    --triage off \
    --llm-provider ollama --llm-model qwen2.5:7b \
    --max-papers 500 \
    --db-path data/mining_models/amyloid_aggregation.db

# 2. export the mining SQLite to the JSON shape amylodb expects
python scripts/export_mined.py \
    --db data/mining_models/amyloid_aggregation.db \
    --out mined.json

# 3. rebuild, reconciling curated + mined
amylodb build --config configs/default.yaml --mined mined.json
```

Use a new `--db-path` for the first model-based run so the earlier database
stays as the baseline for comparison. `pipeline_stats.json` is written next to
the database.

### 4a. Expand the corpus by year bins

One broad query saturates: of the 2906 papers it returned, only 1234 yielded a
protein, and the accumulation curve was still rising (the last 500 papers added
16-20 new proteins), so recall was capped by the per-query limit rather than by
the literature. `--corpus-plan` searches each term group separately over
publication-year bins, so the limit applies per bin, and de-duplicates the PMIDs:

```bash
# counts only, nothing fetched -- see the size before committing
python scripts/plan_corpus.py --email you@example.com --out corpus_plan.json

# then the expanded run
python -m amylodb.mining.pipeline \
    --email you@example.com \
    --mode models --ner-model openbioner --models-dir models --device auto \
    --triage off --llm-provider ollama --llm-model qwen2.5:7b \
    --corpus-plan --date-from 1998/01/01 --bin-years 2 --per-bin 400 \
    --max-papers 20000 \
    --db-path data/mining_v3/amyloid_aggregation.db
```

The groups are in `settings.TARGETED_TERM_GROUPS`: `broad` plus bacterial,
fungal and plant functional amyloids, prion-like and low-complexity domains,
RNP condensates and stress granules, structural evidence, and material
assembly. Restrict them with `--groups broad rnp_condensates`. 1998 is the
floor: the first fibre-diffraction and ssNMR structural data date from then, and
earlier records support no structural tier. `--per-bin` is the budget per group
and bin; `--max-total` caps distinct PMIDs. `plan_corpus.py --estimate-overlap 3`
retrieves the three most recent bins per group to measure how much the groups
duplicate each other before the full fetch.

A long run is expected to lose a few papers: `pipeline_stats.json` reports
`ner_chunks_failed` (chunks the zero-shot NER could not process) and
`extraction_failed` with the `failed_pmids`. Neither ends the run. To retry just
those papers, re-run the same command without `--force` -- processed papers are
skipped, so only the failures are attempted again.

Mine into a new `--db-path`, then re-validate and audit that database as in
sections 5-6a; the role thresholds should be re-calibrated once the expanded
review sheets exist, since they were fitted on the smaller corpus.

Useful switches: `--nli-threshold` / `--nli-margin` (decision rules),
`--no-llm-fallback` (models only), `--triage gate --triage-threshold 0.5`
(drop low-scoring abstracts; only after checking the scores),
`--triage annotate` (score abstracts without dropping any; off by default),
`--no-require-uniprot`, `--no-mention-gate` (not recommended),
`--no-structner`.

## 5. Re-validate an existing mining database (no re-mining)

The precision rules below (section "Precision guards" in `mining.md`) are applied
during mining from now on. A database mined before them is re-checked in place of
a new run: every protein is re-resolved from its written surface forms and the
abstracts it is linked to, without querying PubMed again or re-running the models.

```bash
python scripts/revalidate_mining.py \
    --db  data/mining_v2/amyloid_aggregation.db \
    --out data/mining_v2/amyloid_aggregation.revalidated.db \
    --report revalidation.json
```

* The source database is copied, never modified.
* Rows are not deleted: `proteins.status` becomes `kept` or
  `dropped:<reason>` (`noise`, `noise_abbreviation`, `no_uniprot_match`,
  `short_symbol_uncorroborated`, `off_topic`, `no_aggregation_mention`,
  `regulator_only`, `lookup_error`), so every decision can be inspected.
* `aggregation_evidence` says how an aggregator is supported: `curated`
  (built-in list), `text` (a sentence states that the protein itself
  aggregates; about 3 in 4 right on review) or `model_only` (the NLI vote
  alone, a predicted amyloid; about 1 in 4 confirmed). All three stay in the
  database; the tier is the last column of `records.tsv` and a field of the
  graph nodes, and a `model_only` record votes with half weight. Exclude
  `model_only` when a set of confirmed amyloids is needed (training labels).
* Condensate candidates with the subtype `regulator` reached the list through
  regulator votes only (they regulate a condensate rather than belong to it).
* `dropped:lookup_error` means UniProt could not be reached after four
  attempts with backoff, not that the name has no entry. Re-running the
  re-validation retries these proteins.
* A regulator vote is stored in `role_votes` but makes neither a modifier nor a
  dual role (0 of 16 regulator calls were right on review): such a protein is a
  condensate candidate when its text names a condensate, else
  `dropped:regulator_only`.
* `protein_role` separates aggregators from modifiers (chaperones,
  disaggregases). Modifiers are not amyloid records; the build writes them to
  `modifiers.tsv` with the aggregators named in the same papers.
* The resolved identity goes to `resolved_name` / `resolved_organism`
  (`protein_name` is part of a UNIQUE key); rows that now share a UniProt
  accession are merged by `amylodb build`.
* UniProt is queried once per protein (cached, `--sleep 0.05` between calls);
  about 5 minutes for 300 proteins.
* `--rescore-claims --models-dir models` re-decides the role of every kept
  protein that is not a curated canonical one from its stored abstracts with
  the NLI model: aggregator, or a modifier of subtype chaperone, inhibitor,
  regulator or associated (co-deposits with another protein's amyloid). A
  protein whose abstracts assert none of these is dropped as `no_role_claim`.
  At most `--max-papers 25` abstracts per protein are scored; 8 hypotheses per
  abstract, roughly 10-20 minutes for 200 proteins on MPS. The raw per-abstract
  role scores are stored (`role_scores`) so the decision thresholds can be
  fitted to manual marks afterwards (section 5a).

```bash
python scripts/revalidate_mining.py \
    --db  data/mining_v2/amyloid_aggregation.db \
    --out data/mining_v2/amyloid_aggregation.revalidated.db \
    --report revalidation.json \
    --rescore-claims --models-dir models --device auto
```

### 5b. Check a lookup change on the whole population first

Identity rules change which UniProt entry a name resolves to. Before a full
re-validation with the models, re-resolve the proteins of the last re-validated
database and look at what moves (UniProt only, no NLI; about as long as a
re-validation without `--rescore-claims`):

```bash
python scripts/compare_lookup.py \
    --db data/mining_v3/amyloid_aggregation.revalidated.db --out lookup_diff4 \
    --exclude lookup_diff_review_checked.tsv lookup_diff2_review_checked.tsv lookup_diff3_review_checked.tsv
```

It prints how many proteins keep their accession, change it, lose it or gain
one, by match type, and writes `lookup_diff.tsv` (every change) and
`lookup_diff_review.tsv` (40 changes to mark). A rule is kept when the marked
changes are mostly right.

Each row compares the database (`old_*`: what the last re-validation stored)
with the current code (`new_*`). `change` is `reidentified` (another entry),
`lost` (kept before, dropped now; `new_status` says why), or `gained` (dropped
before, kept now). The comparison is always with the database, not with the
previous diff, so a change stays listed until the database is re-validated;
pass the sheets already marked to `--exclude` so the sample shows new rows
only. Mark `correct` as y when the new state is right: the right entry for a
kept protein, or a justified drop (generic class, several members in the text,
not a protein).

Then export, build and audit the re-validated copy:

```bash
python scripts/export_mined.py \
    --db data/mining_v2/amyloid_aggregation.revalidated.db --out mined.json
amylodb build --config configs/default.yaml --mined mined.json
python scripts/audit_mining.py \
    --db data/mining_v2/amyloid_aggregation.revalidated.db \
    --json audit_revalidated.json --review-sample 50 --review-out review.tsv \
    --review-dropped 30
```

## 6. Independent precision estimate

The `noise names` line of the audit applies the pipeline's own filter to its own
output, so it can only show consistency, not precision. The independent estimate
is a manual review: `--review-sample 50` writes a random sample of kept proteins
(seed fixed) with the UniProt name, organism, role, paper count and one sentence
from the linked abstracts. Fill the `correct` column with `y` (the sentence
supports the protein's identity and the role in the `role` / `subtype`
columns -- an aggregator that forms the aggregates itself, or a modifier of
the stated subtype) or `n`, then:

```bash
python scripts/audit_mining.py --score-review review.tsv
```

The kept sample holds only aggregators -- the exported amyloid records -- so
`correct = y` means the protein forms the aggregates itself. Modifiers are
sampled separately with `--review-modifiers 20` (chaperone, inhibitor,
regulator, associated), where `correct = y` means the identity and the subtype
are right.

**Marking rule.** A role is right when *any* of the protein's papers supports
it, for aggregators and modifiers alike: gelsolin is an aggregator because
papers describe the AGel amyloid, and an inhibitor because others show it
dissolving Abeta fibrils. Each row therefore shows up to three evidence
sentences from different papers (`evidence_1..3`, PMID in brackets), those using
the sheet's vocabulary first, and `supporting_papers` counts the papers that
have such a sentence. `modifier_roles` lists the modifier subtypes some paper
supports, including those of aggregators.

Fill `error_type` for every `n`: `identity` (wrong entry), `role` (an
aggregator that only acts on another protein, or the reverse), `offtopic`
(named but not studied, e.g. an immunostaining marker), `condensate`
(stress-granule / P-body / LLPS literature without aggregation) or `subtype`.
Only `role` errors are used to calibrate the role thresholds.

Outputs of the build besides the amyloid records: `modifiers.tsv` holds every
protein with a modifier role, dual-role aggregators included (`primary_role`
says which); `condensate_candidates.tsv` holds the condensate candidates and
every aggregator or modifier named beside a condensate term
(`condensate_evidence`), the input for condensatedb.

Re-validation routes proteins whose literature never mentions aggregation:
with condensate vocabulary they become `condensate` candidates, written by the
build to `condensate_candidates.tsv` (input for condensatedb); without it they
are dropped as `off_topic`. `--review-condensates 20` samples them.

Each accession appears once in the sample (rows that the build merges count as
one protein). The rejections are checked the same way: `--review-dropped 30`
writes `review_dropped.tsv` from the dropped rows, where `correct = y` means
the rejection was right; `1 - precision` of that sheet estimates the share of
real proteins lost.

This prints precision with a Wilson 95 % interval. With 50 marked rows the
interval half-width is about 0.1; mark 100 for about 0.08.

### 6a. Calibrate the role decision

Zero-shot NLI entails "X forms amyloid fibrils" for almost any protein named in
an aggregation abstract, so the fixed thresholds leave regulators and inhibitors
classed as aggregators. After marking one or more review sheets, fit the
aggregator/modifier thresholds to those marks (`correct = y` = the protein forms
the aggregates itself):

```bash
python scripts/calibrate_roles.py --db data/mining_v2/amyloid_aggregation.revalidated.db \
    --labels review.tsv review2.tsv review3.tsv --min-recall 0.9 --out role_thresholds.json
```

The reported precision is leave-one-out (the thresholds are chosen on the other
rows), so it is not fitted to the rows it scores. `role_thresholds.json` records
the chosen statistic (mean or max over a protein's abstracts), the minimum
self-aggregation score and the minimum lead over the modifier claims.

Feed the file back on the next re-validation to apply the calibrated rule
instead of the per-abstract vote:

```bash
python scripts/revalidate_mining.py --db ... --out ... --rescore-claims \
    --models-dir models --role-thresholds role_thresholds.json
```

## 7. Compare with the baseline

```bash
python scripts/audit_mining.py --db data/mining/amyloid_aggregation.db        --json audit_before.json
python scripts/audit_mining.py --db data/mining_models/amyloid_aggregation.db --json audit_after.json
```

Baseline measured on the first run (444 abstracts, 690 proteins):

| Metric | Baseline |
|--------|----------|
| reversibility filled | 2.75 % |
| aggregate structure filled | 7.7 % |
| functional class filled | 23.9 % |
| template echoes (all fields) | 27 |
| confidence minimum | 0.45 (artefact) |
| noise names | 65 (9.4 %) |
| spurious canonical protein-paper links | 151 / 273 (55.3 %) |

## 8. A quick dry run first

Before a long mining run, check the PubMed query and counts without extraction:

```bash
python -m amylodb.mining.pipeline --email you@example.com --dry-run --max-papers 20
```
