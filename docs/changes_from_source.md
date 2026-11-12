# Changes from the source branches

amylodb is built from two branches of `juliapatsyukova/Amyloid-DB`:

- `merged_version` — the curated aggregation pipeline (`v2/amyloid_pipeline`),
  9 source parsers, 88-feature computation, evidence-weighted consensus.
- `llm-sukhanova` — the literature-mining stack (`src/pubmed`, `src/ner`,
  `src/llm`, `src/hybrid`, `src/string_db`, `src/database`).

The curated package is vendored under `src/amylodb/curated/` and the mining
stack under `src/amylodb/mining/`. The reconciliation layer, unified schema,
provenance/evidence model, feature matrix and graph export are new.

Four substantive changes were made relative to the source code, each for a
stated reason rather than style.

## 1. Two schemas reconciled, not merged

The curated line keys records on peptide/region identity; the mining line keys
on protein identity. A single dedup key would either fail to join them or
collapse distinct biology. `AmyloidRecord.dedup_key()` is **level-aware**: a
`PROTEIN` record and a `REGION` record can never fall into the same group, so
reconciliation is well defined across origins. Overlapping records are resolved
by tier-weighted voting (a dominant label must out-weigh the runner-up by ×1.5,
otherwise the group is reported as an unresolved conflict rather than guessed).

## 2. Prion handling: keep prion-like domains

The mining branch excluded prions and prion-like proteins wholesale, by
`EXCLUSION_TERMS` at the PubMed query and by a substring check
(`"prp" in name`) in the rule-based filter. Two problems:

- **Substring false positives** — `"prp" in name` matched `PRPF8`, `PRPF40A`,
  `PRPS1`; `"prion-like"` was caught by the `"prion"` exclusion.
- **Wrong for this project** — translation-coupled RNP condensates are organised
  by prion-like-domain RBPs (FUS, hnRNPA1, TDP-43, PRRC2A/B). Excluding
  "prion-like" removes the core of the sample.

`amylodb.provenance.classify_prion` uses word-boundary patterns and a three-way
`PrionClass`: prion-like-domain records are kept and tagged; only transmissible
mammalian prion pathology (PrP^Sc, scrapie, CJD, BSE) is a candidate for
exclusion, controlled by `reconcile(exclude_transmissible_prion=...)`. The
query-level `EXCLUSION_TERMS` was narrowed to the transmissible set.

## 3. Fixed-width feature matrix

The curated TSV export took its column set from the first record and wrote with
`extrasaction="ignore"`, so the dipeptide block (computed as the top-25 *per
record*) was reduced to whatever dipeptides the first record happened to have —
three columns in the shipped `consensus_unified.tsv`, the rest silently dropped.
`amylodb.features_matrix` replaces this with a deterministic **444-wide** matrix:
21 physicochemical + 20 single-AA + the full 400 dipeptides + 3 grouped
fractions. The column order depends only on the amino-acid alphabet, so two
builds of the same sequences are byte-identical.

## 4. Amylobase records recovered

Amylobase ships column names with trailing spaces (`"Protein ID "`), so
`row.get("Protein ID")` always missed and every record was dropped; the curated
`is_valid()` also required a sequence or a PDB id, which Amylobase records
(UniProt + coordinates, no residue string) do not carry. Both are fixed: the
parser strips field names, and `is_valid()` accepts a located region
(UniProt + coordinates) whose sequence is fetched downstream. This recovers the
kinetic source (23 records in the current curated build).

## 5. Mining precision

Measured on the first mining run (444 abstracts, 690 stored proteins) with
`scripts/audit_mining.py`, and corrected:

| Defect | Measured | Correction |
|--------|----------|------------|
| Normaliser matched a name *inside* a synonym, so generic words ("amyloid") became amyloid-beta and short tokens attached to the first synonym containing them; the canonical's UniProt id was then assigned | 151 of 273 canonical protein-paper links (55.3 %) point to abstracts that never name the protein; amyloid-beta 77/87, insulin 33/58 | exact match; forward partial match only for synonyms >= 4 characters with strict boundaries; 3-letter gene symbols only as standalone tokens; plus a mention gate before linking |
| Pattern confidence compared absent keys with `'unknown'` (always unequal) | every protein received +0.45; minimum confidence exactly 0.45; 74 % passed the LLM threshold, so the LLM was rarely called | absent or undecided fields add nothing |
| LLM categorical answers were not validated | 27 stored values were echoes of the option list (`beta-sheet\|alpha-helical\|mixed\|unknown`) | answers outside the vocabulary count as no answer; prompts ask for exactly one value |
| Methods, diseases, reagents and generic phrases accepted as proteins | 65 names (9.4 %), e.g. `tem`, `tht`, `egcg`, `als`, `abnormal protein` | noise filter; methods from the structure NER are purged from candidates |
| Prion-like abstracts excluded by the prefilter and LLM prompt (the query had been narrowed, these two had not) | abstracts mentioning "prion-like" never reached extraction | prefilter, prompt and response parsing follow the same policy as `classify_prion` |
| A decided value could be overwritten by `unknown` from a later paper | -- | updates never replace a decision with a non-decision; per-paper evidence kept in `protein_claims` |

Reversibility -- the target variable of the dissertation -- was filled for
2.75 % of mined proteins; this is what the model-based extractor addresses.

## 6. Model-based extraction (`--mode models`)

Fields are decided by zero-shot NLI over the sentences that mention each
protein, with explicit thresholds (entailment >= 0.7, lead over the opposing
claim >= 0.2), instead of keyword counting. Decisions that deserve a stated
basis:

* **Absence is not negation.** LLPS is set only on entailment; a low score
  leaves it `unknown`, never `no`.
* **Both reversibility claims entailed -> `conditionally_reversible`.** This is
  the case of dynamic droplets that mature into solid aggregates. Across papers,
  a class with >= 2/3 of per-paper decisions wins; reversible and irreversible
  reports without such a majority also give `conditionally_reversible`, since
  different papers usually describe different conditions or variants.
* **Reported evidence is not verified evidence.** A mined record stays at tier
  `TEXT_MINED` even if its paper reports cryo-EM; the strongest reported method
  is kept as `reported_method_tier` for the graded classification to use
  explicitly.
* **The triage model is annotate-only by default.** BiomedBERT-Amyloid was
  trained to recognise articles on antibody effects on amyloid formation, a
  narrower notion than "reports aggregation"; its score is stored, and gating on
  it is a separate decision to make after inspecting the scores.
* **UniProt resolution is not restricted to human**, so bacterial and fungal
  functional amyloids are not lost to the UniProt requirement.


## 7. Structure-NER token limit and the chaperone discriminator

* The PubMedBERT structure-NER backbone has a 512-position limit and declares no
  tokenizer max length, so a long char-chunk overflowed its position embeddings
  (RuntimeError: tensor a (592) vs b (512)). Chunks are now budgeted in tokens
  against the model's own tokenizer, with original-text offsets preserved and a
  model_max_length guard as a backstop.
* A disaggregase claim was added to the NLI set. A chaperone such as Hsp70 or
  Hsp104 appears in an aggregation abstract because it dissolves aggregates, not
  because it forms them; when the disaggregation claim dominates the aggregation
  claim, the aggregator call is withheld, so the protein does not become a
  record on that basis.

## 8. Re-validation of mined names

Manual review of the v2 run (314 proteins) found roughly a quarter of the names
questionable: reagents (urea, DMSO, CTAB), small molecules hidden behind
acronyms that UniProt resolves to real proteins (PDI = perylene diimide,
OleA = oleuropein aglycone, SCL = sclerotiorin, GAL = gallein, CSA = chondroitin
sulfate A), domains and fragments (NAC domain, BRICHOS domain, protein-43),
class phrases (transcription factors, antimicrobial peptides) and chaperones
recorded as aggregators.

* The rules in `mining.md` ("Precision guards") now run inside the mining
  pipeline and in `scripts/revalidate_mining.py`, which applies them to an
  existing database without re-mining.
* On a copy of the v2 database (UniProt answers taken from the stored
  accessions, so only the name rules were exercised) 254 of 314 proteins were
  kept, 51 dropped as noise and 9 as abbreviations of non-proteins
  (amyloidogenic peptides, full-length amyloid fibrils, prion-like domains,
  amyloid-like fibril, amyloidogenic core region, cyclic peptides, lysine
  residue, olfactomedin domain, cerebral amyloid angiopathy). Of the dropped
  names none is a protein that forms or modulates amyloid in the cited abstract.
  The live run additionally re-queries UniProt with the new match rules, which
  can drop further rows as `no_uniprot_match` or `short_symbol_uncorroborated`.
* The audit separates the consistency check (the pipeline's own filter on its own
  output) from an independent estimate: a seeded random review sample scored by
  hand, reported with a Wilson 95 % interval.
* BiomedBERT-Amyloid triage is off by default (0.34 % of abstracts of an
  aggregation-specific query scored >= 0.5).

### 8.1 First live re-validation and manual review

The live run (UniProt queried) kept 200 of 314 proteins. A manual review of 50
kept proteins gave precision 0.74 (Wilson 95 % CI 0.60-0.84). Of the 13 rows
marked wrong, 8 were regulators or inhibitors of another protein's
aggregation recorded as aggregators (artemin, ATF6, SIRT1, SIRT2, Sel1L,
cathepsin B, FAIM, Btn2), 3 were identity errors (FAP resolved through the
short form to prolyl endopeptidase; LBP from a rat study resolved to human,
sampled twice), and 2 were chaperones. Among rows marked right, several were
identity errors from gene-only matches in organisms the abstract does not name
(Glu8 = a residue of beta-endorphin resolved to rice endoglucanase; DHAA = a
dehydroamino acid resolved to a Paracoccus aldolase; FadA of Fusobacterium
resolved to E. coli FadA; RepA resolved to a rice protease) and PRC1 the
Polycomb complex resolved to the cytokinesis regulator. 52 rows were dropped
as `no_uniprot_match`, among them real proteins (p53, fibronectin, SOD1,
glucagon, calmodulin, human serum albumin, YB-1, Hsc70).

Corrections: Swiss-Prot-only search with gene synonyms, alternative and
product names and qualifier-stripped retries; organism-in-abstract
requirement for non-human gene-only matches; species words in the name select
the species; residue notation, "complex" heads and source-qualified class
phrases are noise; reversed abbreviation definitions; the short-form fallback
is gated; the review sample has one row per accession and a dropped-row sheet
measures losses.

Not yet corrected: regulators recorded as aggregators. This needs an NLI claim
that separates "X forms aggregates" from "X modulates the aggregation of
another protein", i.e. re-scoring the stored abstracts with the models.

### 8.2 Second review and role re-scoring

After the UniProt corrections the live re-validation kept 214 of 314 proteins
(181 distinct accessions). The review of 50 kept proteins gave precision 0.72
(95 % CI 0.58-0.83); the review of 30 dropped proteins found 21 rejections
right. Remaining errors among kept proteins: regulators, inhibitors and
degrading enzymes recorded as aggregators (TRIM11, Sel1L, Cdh1, ROCK2,
trypsin, metallothionein-3) and co-deposited proteins (serum amyloid P);
identity errors through qualifier-stripped gene matches (Gla protein -> GLA),
"protein + letter" names (protein C -> prefoldin 4), partial runs over
alternative names (FABP -> GOT2) and antibody loops (CDR1 -> gene CDR1).
Among dropped rows: a curated canonical (Aβ42) rejected by the short-form
gate, "tau variant" rejected as noise, and Hfq rejected although E. coli is
named.

Corrections: four NLI modifier claims and `decide_role`; `--rescore-claims`
in the re-validation; qualifier-stripped queries may not match genes; runs
need a specific word and only the recommended name; construct words are
stripped ("tau variant" -> "tau"); CDR loops are noise; curated canonicals
bypass the gates; a short symbol in a non-human organism named in the text
passes; common crop names count as organism mentions.


### 8.3 Third review: role re-scoring was too weak

With `--rescore-claims` on, the live run kept 207 of 314 proteins but changed
only 19 roles (18 chaperones, 1 inhibitor) and dropped 3 as `no_role_claim`;
precision stayed at 0.72. The stored votes explain it: zero-shot NLI returned
312 aggregator votes against 2 modifier votes, entailing "X forms amyloid
fibrils" for regulators (ATF6, SIRT3), enzymes (cathepsin B) and a transcription
factor (HSF1) named in an aggregation abstract. NLI cannot, on its own, separate
the protein that aggregates from the one that acts on another protein's
aggregation.

Two responses:
* The per-abstract role scores are now stored (`role_scores`), and
  `scripts/calibrate_roles.py` fits the aggregator/modifier thresholds to the
  manual marks already collected, reporting a leave-one-out precision. This
  replaces guessed thresholds with ones measured against review.
* Identity fixes from this review: UniProt short names are treated as symbols
  (PAc, HSP27 no longer match by name); a symbol with a digit passes the
  short-symbol gate (YB-1, PS1) while a letter-only one still needs
  corroboration (Kap, Lap); "paired helical filament" and other `filament`
  heads are noise; HSF1 and other `...factor` names default to the regulator
  subtype; and a name truncated before its head noun is completed from the
  abstract (Prostatic Acid -> Prostatic Acid Phosphatase, acyl-coenzyme A ->
  acyl-coenzyme A thioesterase-7).

Still open: FadA (Fusobacterium, absent from Swiss-Prot) and the Nipah W protein
remain dropped; the regulator/aggregator split now depends on the calibrated
thresholds, to be confirmed on the next marked sheet.

### 8.4 Held-out confirmation and a lexical modifier override

With the calibrated thresholds applied, a fresh review sheet (review5, held
out from calibration) gave aggregator precision 0.78 (95 % CI 0.65-0.87) and 39
correctly separated modifiers (17 chaperone, 12 associated, 6 regulator, 3
inhibitor). Every one of the 11 wrong rows is the same class -- a protein that
acts on another protein's aggregation recorded as an aggregator (neuropilin-1,
metallothionein-3, Sel1L, cathepsin B/C, midkine, Btn2, ELP3, OASL) -- with the
correct identity. None is an identity error and none a regression, confirming
that the aggregator/modifier confusion is the ceiling of the text-mined tier and
that identity resolution is sound.

Several of these name the protein with an explicit modifier noun the NLI model
ignored, so a deterministic override now flips an aggregator to a modifier when
a chaperone-class noun (chaperone, sequestrase, disaggregase, holdase, foldase)
shares a mention sentence with it. The check is nouns only -- these never
describe an amyloid-forming protein, so it cannot flip a true aggregator -- and
runs regardless of the model, catching cases the UniProt `Chaperone` keyword
misses (metallothionein-3, Btn2). Verb cues ("inhibits", "attenuates") are left
to the NLI role claims, since attributing them to the right subject needs
parsing the override deliberately avoids.

## 9. Corpus expansion by term group and year bin

The source pipeline issued one query built from `AGGREGATION_TERMS` with a
single `retmax`. Measured on the v2 corpus that saturates: 2906 papers, only
1234 of which yielded a protein, and a still-rising accumulation curve (random
subsets of 500, 1000, ... 2906 papers gave 50-63, 87-111, ... 181 distinct
accessions, the last 500 adding 16-20). The limit was the query budget, not the
literature.

* `TARGETED_TERM_GROUPS` names six classes the broad query under-covers:
  bacterial functional amyloids (curli, Fap, TasA, chaplins, PSMs), fungal and
  plant ones (hydrophobins, HET-s, seed storage proteins), prion-like and
  low-complexity domains, RNP condensates and stress granules, structural
  evidence (cryo-EM, ssNMR, polymorphs), and material assembly. Each is
  searched on its own budget rather than competing inside one query.
* `PubMedSearcher.year_bins` / `search_corpus` slice every group by publication
  year (default two-year bins from 1998, the first fibre-diffraction and ssNMR
  structural data), so the per-query cap applies per bin. PMIDs are
  de-duplicated across groups and bins in discovery order, so a `max_total` cut
  keeps whole early bins instead of a random subset.
* `PubMedSearcher.count` and `scripts/plan_corpus.py` report hit counts per
  group and bin without fetching anything, with `--estimate-overlap` to measure
  how much the groups duplicate each other before the full run.
* The pipeline gains `--corpus-plan`, `--groups`, `--bin-years`, `--per-bin`
  and `--max-total`; without `--corpus-plan` the single-query behaviour is
  unchanged.

The role thresholds in `role_thresholds.json` were fitted on the v2 corpus and
should be re-calibrated from review sheets drawn on the expanded database.

### 9.1 Roles must not be decided while mining

The first expanded run (12873 papers searched, 9617 stored, 1870 proteins)
exposed an error introduced with the modifier claims: `decide` acted on all four
of them per paper with the fixed 0.7 / 0.2 thresholds. Zero-shot NLI entails
"regulates the expression, processing, modification or clearance of an
aggregating protein" for almost anything named in an aggregation abstract, so
1220 of 1870 proteins were stored as modifiers (1073 as `regulator`), and
because `forms_aggregates` was computed as "aggregates and not a modifier", the
primary field was suppressed for 1202 of them -- amyloid-beta was recorded as an
`inhibitor`, insulin and apolipoprotein E as `associated`.

The cause is architectural, not a threshold value: one paper's scores cannot
separate a protein that aggregates from one that acts on another protein's
aggregation, and the mining pass sees one paper at a time. So the decision moved
to where the evidence is:

* `decide` withholds the aggregator call only for a dominant disaggregation
  claim, as it originally did; the other role scores go into `role_scores`.
* `assign_role` takes `from_claims`, and the mining pipeline passes
  `from_claims=False`: deterministic modifier evidence only.
* `role_scores` is stored at mining time, so thresholds can be calibrated from
  an existing database without re-mining.
* `revalidate_mining.py` repairs `forms_aggregates` for every protein whose role
  it re-decides, and reports `forms_aggregates_repaired`.
* `_print_statistics` now prints the per-gate drops, the failed papers and the
  skipped NER chunks, so a run's losses are visible without the stats file.

A database mined before this fix is recovered by re-validation; the search and
the model inference do not need repeating.


### 9.2 Review of the expanded corpus: identity becomes the main error

Re-validation of the 12873-paper corpus kept 1786 of 1870 proteins. The marked
sheets, re-checked row by row against UniProt and the evidence sentence, gave:

| sheet | marked precision | after re-check | 95 % CI |
|---|---|---|---|
| aggregators (50) | 0.74 | 0.58 | 0.44-0.71 |
| modifiers (20) | 0.85 | 0.20 | 0.08-0.42 |
| dropped, rejection right (30) | 0.60 | 0.53 | 0.36-0.70 |

Of 21 wrong aggregators, 11 were identity errors -- a symbol resolved to the
wrong entry: yeast Rim4 to human RIMS4 in a paper that says "in yeast",
H. pylori CagA to S100A8 (synonym CAGA), TORC1 to CRTC1, the tau hexapeptide
PHF6 to PHD finger protein 6, Bacillus CalY to calcyon, TPPP/p25 to TMED9,
"LF-MF-treated PG (MPG)" to the MPG glycosylase. The expanded queries name many
more short symbols, and many collide with human gene synonyms. Of the modifiers,
5 were identity errors, and 9 were stress-granule or P-body proteins whose
abstracts never mention aggregation (G3BP1, ZAP, Roquin, Raptor). Roughly half of
the `no_role_claim` drops were real proteins (vitronectin, whose sentence says it
forms amyloid fibrils; ubiquilin-2, nucleobindin-1, fibrinogen).

A rule that looked right on the sheets was measured on the whole base and
rejected. "A gene-synonym match must be corroborated by the UniProt name in the
abstract" caught 7 of 16 identity errors and lost no correct row among the 70
reviewed; but a random sample of the 158 proteins it would drop showed about two
thirds to be correct synonym matches -- Slug (SNAI2), HRD1 (synoviolin), TrkB,
MDR1, HSJ1, YB-1, Vasa, Bri2, HspB5, lactoferrin, MT-3 -- because authors write
the synonym, not the systematic name. The discriminating property is not
"synonym" but ambiguity: several unrelated entries sharing one symbol.

Changes:
* UniProt selection: a primary gene or a recommended name beats a synonym; a
  synonym carried by one entry is accepted, one shared by several is taken
  only when the abstract corroborates exactly one; an entry of an organism the
  abstract names beats a human synonym, and in an abstract naming only another
  organism a human gene-symbol hit needs corroboration. Other organisms are
  searched when the abstract names one or the human match is a synonym only.
* Corroboration ignores generic words ("alpha", "membrane", "factor", ...) and
  participles: "alpha" had corroborated SCN2A for alpha-haemoglobin.
* Family and enzyme-class names (importin, AAA+ ATPase, PI3K), named complexes
  (TORC1, mTORC1), chemical heads (polysaccharide, lipid) and sample
  descriptions ("X-treated Y") are noise.
* Size and segment labels: p62, p97, p38, p65, p53, p300, p21, p27, p16, p35
  and PHF6 are searched as the gene the proteostasis literature means (SQSTM1,
  VCP, MAPK14, RELA, ..., MAPT); any other size label (p25, p50, p110) must be
  corroborated. Resolved literally, p97 had become Gem-associated protein 4,
  p38 CRK and p65 synaptotagmin-1.
* Aβ / APP name variants (hyphenated, plural, "Abeta-protein precursor") map to
  the curated canonical; they had been dropped as `no_uniprot_match`.
* Relevance routing in re-validation: a protein whose abstracts contain no
  aggregation term becomes a `condensate` candidate when they contain condensate
  terms (written to `condensate_candidates.tsv`, input for condensatedb) and is
  dropped as `off_topic` otherwise; an aggregator must share a sentence with an
  aggregation term (`no_aggregation_mention`). On the v3 base, 457 of 1180
  modifiers had no aggregation term in any abstract.
* An NLI abstention no longer drops a protein; the vocabulary gates decide.
* The chaperone-noun override requires the noun applied to the protein
  (apposition, copula, possessed activity) and never acts against an
  aggregator majority. Accepting any shared sentence had turned Sup35, with 15
  aggregator votes, into a chaperone ("Hsp104 is a chaperone that dissolves
  Sup35 fibrils").
* Re-validation no longer decides a role from the stored per-paper claim scores
  with fixed thresholds; that path had survived there after being removed from
  mining. A role from claims comes only from re-scoring.
* Review sheets carry `error_type` (identity | role | offtopic | condensate |
  subtype); `calibrate_roles.py` learns from role errors only.

A dry run on the v3 base with stored identities and no model removed 9 of the
wrong aggregators at the cost of one correct row (GAPR-1, whose sentence does not
say it aggregates) and raised precision on the remaining reviewed aggregators
from 0.58 to 0.70. The UniProt selection rules need the live service and are
tested against recorded-shape responses only.

### 9.3 Decisions on scope and dual roles

Two decisions taken after the review:

* Stress granules, P-bodies and the nucleolus define a separate class of
  candidates for condensatedb, not modifiers of amyloid. Proteins whose
  literature has no aggregation term are routed there; proteins with an amyloid
  role that are named beside a condensate term (TDP-43, FUS) are listed there
  as well, through `condensate_evidence`.
* A role holds when any of a protein's papers supports it, for aggregators and
  modifiers alike. Roles therefore became a set: the primary `protein_role`
  plus `modifier_roles`, filled from per-paper votes; a dual-role protein is
  both an amyloid record and a row of `modifiers.tsv`. On the v3 base gelsolin
  is the clear case: of its ten papers, several describe AGel amyloidogenesis
  and several its anti-amyloidogenic action on Abeta.

Implementing the second decision exposed a fault in re-validation: role fields
left empty by a new decision were not written, so the mining-time subtypes
survived -- 425 condensate candidates and 370 aggregators of the v3 base still
carried "regulator", which is why the v3 audit counted 1063 regulators where the
re-validation report had 941. The role fields are now overwritten every time.

Review sheets show up to three evidence sentences from different papers and the
number of papers supporting the sheet's role, so the any-paper rule can be
applied without opening PubMed.


### 9.4 Second review of the expanded corpus: family names, lost symbols, regulators

Four sheets were marked after the dual-role run: 50 aggregators, 20 condensate
candidates, 30 modifiers and 30 dropped proteins. Scored with error types
(aggregators by the curator, the rest checked against the sentences):

| sheet | precision | 95 % CI | main errors |
|---|---|---|---|
| aggregators | 0.48 | 0.35-0.61 | identity 12, role 11 |
| condensate candidates (as components) | 0.30 | 0.15-0.52 | role 7 (regulators of granule formation), off-topic 5 |
| modifiers | 0.20 | 0.10-0.37 | identity 7, off-topic 6, role 5 |
| dropped (rejection right) | 0.73 | 0.56-0.86 | lost condensate and functional-amyloid proteins |

By match type, run matches were right in 3 of 19 cases, against 14 of 34 for
gene and 13 of 27 for name matches. Of 277 kept run matches, 220 add a paralog
mark to the written name: "carbonic anhydrase" -> CA6, "tubulin" -> the delta
chain, "apolipoprotein" -> ApoC-I, "transportin" -> transportin-3, "lactate
dehydrogenase" -> the mitochondrial D-LDH. Changes:

* **Family names.** A run match that adds a paralog mark (number, letter, Greek
  letter, "mitochondrial") is kept only when it is the only such entry of its
  organism among the hits (perfringolysin O, synphilin-1, TDP-43); a family
  name with several members is unresolved. A run adding "like" or "homolog" is
  another protein (ataxin-3 had resolved to ATXN3L, VHL to VHLL, ataxin-1/2 to
  their "-like" paralogs). Hits are ranked by identity before selection --
  recommended name, primary gene, run, synonym -- so an exact name further down
  UniProt's list wins over a run listed first. Names that UniProt naming cannot
  reach or that the family rule leaves ambiguous are mapped by alias (BACE1,
  BCHE, TNF, FBL, STAU1, EIF2AK3, DDX3X, G3BP1, EWSR1, HSPB1, CALM1).
* **Lost symbols in model organisms.** 26 of 46 `no_uniprot_match` drops were new
  and came from the rule that a human gene hit in an abstract naming only a
  non-human organism needs corroboration: AKT1 and PIK3CA in C. elegans papers,
  Cdc42 in zebrafish, Drosha and Dvl2 in flies. A first fix accepted the human
  primary symbol for such papers; review rejected it, and section 9.5 replaces
  it with a search in the named organism itself. Letters+digits names also
  search the hyphenated gene ("akt-1", "lin-28").
* **UniProt failures.** A failed request was read as "no hits" and the empty
  result cached. Requests now retry with backoff on 429, 5xx and network
  errors; a persistent failure raises `UniProtUnavailable`, is not cached, and
  gives the status `dropped:lookup_error` in re-validation (the mining pipeline
  keeps such a protein unresolved for re-validation). No live failure was
  proven in this run -- the context check above explains most new drops -- but
  the old path could not tell the two apart.
* **Mentions.** The mention test behind `no_aggregation_mention` was exact; it
  now also accepts spelling variants (Greek letters spelled out, optional
  hyphens and parentheses, plural): beta(2)-microglobulin, class I
  hydrophobins and human stefin B were dropped by it. On the population the
  change rescues 7 proteins and removes none.
* **Vocabulary.** "curli" joins the aggregation terms (Cra, CsgD, CsgF were
  off-topic); "polar granules" and "germ plasm" join the condensate terms
  (Vasa, ME31B). "Bovine", "porcine", "equine", "ovine" and "chicken" are
  non-human cues, but "bovine serum (albumin)" as a reagent names no organism.
* **Regulators.** On the modifier sheet the regulator subtype was right in 0 of
  16 cases (421 kept modifiers, 135 aggregators tagged dual-role by it). A
  regulator vote is recorded but no longer makes a modifier or a dual role: a
  protein left without a role becomes a condensate candidate when its text
  names a condensate, otherwise `dropped:regulator_only` (recoverable once a
  calibrated rule exists).
* **One accession, one role.** Rows are per written name, so TDP-43 was an
  aggregator under one name and a modifier under another. Export now gives all
  rows of an accession the same role: aggregator if any row is, else modifier,
  else condensate; modifier roles and condensate evidence are pooled.

`scripts/compare_lookup.py` re-resolves the kept and lookup-failed proteins of a
re-validated database without the NLI models and lists what changed, with a
review sample: the population check for these identity rules, which need the
live service and are tested here against recorded-shape responses only.

### 9.5 Review of the lookup diff

`compare_lookup.py` on the v3 base was followed by a review of 40 changed
resolutions (y = the new state is right: the right entry, or a right drop).
21 of 40 were right; of the 19 wrong, 15 are addressed below and 4 cannot be
solved by lookup (an AL light chain, a transthyretin-like domain of
carboxypeptidase D, a cut "cyclin-dependent kinase inhibitor", actin in a
paper on both yeast and human cells).

* **The human ortholog in a model-organism paper is wrong.** Section 9.4
  accepted the human primary symbol when the organism had no matching entry;
  the review rejected it (Drosha in a fly paper is the fly protein). The rule
  is withdrawn. Instead, when an abstract names only one model organism and
  none of its entries was among the hits, that organism is searched alone,
  Swiss-Prot first and then TrEMBL, by exact name or primary gene: fly Drosha
  has only the TrEMBL entry Q7KNF1. Such hits carry `lookup_stage`
  `taxon_reviewed` or `unreviewed_named`.
* **Truncated family heads.** When a family head does not resolve and the text
  writes it with exactly one member mark ("syntaxin 6-positive"), the lookup
  is repeated with that member; several members (cathepsin B, D, L, S) leave it
  unresolved. A final Roman numeral is also searched in UniProt's Arabic form
  (semenogelin I -> Semenogelin-1).
* **Strains of one species** count as one organism in the family test (PSM
  alpha-2 and alpha-3 came from two S. aureus strains).
* **Classes are noise**: "aspartic protease", "heat shock", "heat-shock
  proteins" (May1 of Cryptococcus for the class of cathepsin D, E. coli Hsp15
  for "heat shock"); a truncated name is no longer extended into a class phrase.
* **Aliases** apply after a leading "human" ("human beta-secretase" had lost
  BACE1) and cover VHL, YAP1, BiP (HSPA5), ACYP2, PTBP1, VEGFA, PADI4.
