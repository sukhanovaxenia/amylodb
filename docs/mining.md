# Literature mining

The mining stage grows the database beyond the curated sources (~2,700 distinct
sequences, of which ~518 are distinct UniProt proteins) by extracting
aggregation and phase-separation claims from PubMed abstracts. It needs network
access (PubMed, UniProt, model downloads on first use), so it runs on your
machine, not inside the core build container.

## Extraction modes

| Mode | What decides the fields | LLM |
|------|-------------------------|-----|
| `models` (recommended) | NER + protein-structure NER + zero-shot NLI | only for fields NLI left undecided |
| `hybrid` | NER + keyword patterns | when pattern confidence < threshold |
| `ner_only` | NER + keyword patterns | never |
| `llm_only` | LLM reads the whole abstract | always |

## Models (`--mode models`)

All are frozen pretrained models used for inference only; none is trained here.

| Role | Model | What it contributes |
|------|-------|---------------------|
| NER | OpenBioNER-v2 (`disi-unibo-nlp/openbioner-base-v2`, via zshot) | proteins / peptides kept apart from methods, diseases, reagents, condensates by described entity types |
| Methods | PDBEurope ProteinStructure-NER v3.1 | experimental methods mapped to universal names and an evidence tier; methods purged from protein candidates |
| Claims | mDeBERTa-v3-base-mnli-xnli (zero-shot NLI) | forms aggregates, reversibility, structure, functional vs pathological, LLPS |
| Triage | BiomedBERT-Amyloid | relevance score per abstract (`annotate` by default: stored, nothing dropped) |
| Gaps | LLM (Ollama / Anthropic / OpenAI) | only fields NLI could not decide |

### How claims become fields

For each candidate protein the sentences that mention it are the premise, and
eight hypotheses are scored for entailment ("<protein> forms amyloid fibrils or
protein aggregates.", "<protein> assemblies are reversible and can dissolve.",
...). With the default threshold 0.7 and margin 0.2:

* a field is decided only when its claim is entailed (>= 0.7) and leads the
  opposing claim by >= 0.2; otherwise it stays `unknown`;
* both reversibility claims entailed -> `conditionally_reversible`; both
  structure claims -> `mixed`; functional and pathological -> `both`;
* LLPS is set to `yes` on entailment only, never to `no` (an abstract that does
  not mention phase separation is not evidence against it);
* a candidate becomes a record only if aggregation or LLPS is entailed;
* `extraction_confidence` is the mean of the scores backing the decisions made.

Per-paper decisions and claim scores are stored in the `protein_claims` table,
so evidence accumulates across papers. At export, protein-level reversibility is
the class holding >= 2/3 of the per-paper decisions; reversible and irreversible
reports without such a majority give `conditionally_reversible`.

Mined records stay at evidence tier `TEXT_MINED` even when their papers report
structural methods; the strongest reported method is kept separately as
`reported_method_tier`.

## Precision guards (all modes)

* **Normaliser**: exact synonym match, plus forward partial match for synonyms
  of >= 4 characters with strict boundaries, plus 3-letter gene symbols only as
  standalone tokens. Names are never matched *inside* a synonym.
* **Mention gate**: a protein is linked to a paper only if the abstract names it
  (canonical name, synonym or gene symbol, boundary-delimited).
* **UniProt requirement** (default on in `models` mode): unresolved names are
  dropped. Lookup tries human first, then reviewed entries of any organism, so
  bacterial and fungal functional amyloids (CsgA, Sup35, HET-s) resolve.
* **Noise filter**: methods, disease abbreviations, reagents and generic phrases
  are not accepted as protein names.
* **Name cleaning**: unbalanced parentheses, possessives and edge punctuation
  are removed before any lookup ("tau (" -> "tau", "FUS's" -> "FUS").
* **Non-protein lists**: chemicals and reagents (urea, DMSO, CTAB, SDS, GM1,
  heparan sulfate), disease heads, class phrases ("transcription factors",
  "antimicrobial peptides", "class I") and part heads (domain, region, motif,
  fragment, fibril, residue, complex, filament) are rejected, as are residue
  notations (Glu8, Lys-1), CDR loops and class phrases qualified by a source
  ("functional amyloid in Pseudomonas"). A UniProt short name ("PAC", "HSP27")
  counts as a symbol, not a name, so it falls under the gene-only gates.
* **A failing model costs one chunk**: zshot 0.0.11 raises `KeyError(-1)` in
  its SMXM span decoder for inputs that cannot be predicted from the text, so
  OpenBioNER is called per ~1000-character chunk inside a guard. A chunk that
  fails is counted (`ner_chunks_failed`) and skipped; the rest of the abstract
  is still annotated, and the PDBEurope span extractor is unaffected. Control
  codes and zero-width marks are replaced by spaces first -- length-preserving,
  so reported spans still index the input, and a tab cannot merge two words.
* **A failing paper costs one paper**: the pipeline wraps each article's
  extraction, counts `extraction_failed` and records the PMIDs in
  `pipeline_stats.json`. Re-running without `--force` retries only those, since
  processed papers are skipped.
* **Truncated names**: a name NER cut before its head noun ("Prostatic Acid",
  "acyl-coenzyme A") is completed from the abstract to the first head noun
  before lookup ("Prostatic Acid Phosphatase"). Fragment patterns ("protein-43",
  "b-a") are rejected; hyphenated symbols with a digit (DJ-1, YB-1) are kept.
* **Abbreviation expansion** (Schwartz-Hearst, both "long form (SF)" and
  "SF (long form)"): a short form is replaced by the long form the abstract
  defines, before lookup. An abstract that defines PDI as
  perylene diimide therefore never resolves PDI to protein disulfide-isomerase.
  A listed non-protein acronym stays noise even when it is spelled out.
* **UniProt match type**: a hit is accepted as `gene` (gene symbol), `name`
  (recommended name) or `run` (one name is a contiguous run of the other and
  every extra word is a qualifier: number, isoform, species or tissue word).
  A new head noun is never a qualifier ("dimethyl sulfoxide" does not match
  "Dimethyl sulfoxide reductase").
* **Family names**: a run that adds a paralog mark (number, letter, Greek
  letter, "mitochondrial") is accepted only when it is the only such entry of
  its organism among the hits; "carbonic anhydrase" or "tubulin" stays
  unresolved, "synphilin" -> synphilin-1 is kept. A run adding "like" or
  "homolog" names another protein and is rejected. Hits are ranked before
  selection: recommended name, primary gene, run, synonym.
* **UniProt availability**: requests retry with backoff on 429, 5xx and
  network errors; a persistent failure is reported as `lookup_error`, never as
  "no match", and is not cached.
* **UniProt search**: Swiss-Prot only (unreviewed fragments otherwise crowd
  the real entry out of the top hits), 25 hits per query, gene synonyms
  (P53 -> TP53, HSC70 -> HSPA8), alternative names (Superoxide dismutase 1)
  and processed products (Glucagon in Pro-glucagon) all count as names. A
  query that finds nothing is retried without a leading "protein" and without
  qualifier words ("human serum albumin" -> "albumin"). A name carrying a
  species word ("equine skeletal myoglobin") resolves to that species.
* **Organism in the abstract**: a gene-symbol-only match in a non-human
  organism is accepted only if the abstract names that organism (genus,
  "E. coli"-style binomial, or a common name such as yeast or mice). This
  rejects Glu8 -> rice endoglucanase and FadA -> E. coli thiolase in a paper on
  Fusobacterium, and accepts Hfq when E. coli is named.
* **Short-form fallback**: when the long form fails and its short form
  resolves, the hit is kept only if the UniProt name is corroborated in the
  text or the short form is the entry's gene symbol (Bcl-2 = BCL2).
* **Short symbols** (<= 3 characters) resolved by gene symbol need
  corroboration: a content word of the UniProt name must appear in the abstract.
* **Syntactic role, not entailment**: zero-shot NLI entails "X forms amyloid
  fibrils" for almost any protein named in an aggregation abstract (the first
  re-scoring voted aggregator 312 times, modifier twice), so the per-abstract
  role scores feed the decision only through thresholds calibrated against
  manual marks (`scripts/calibrate_roles.py`); a UniProt `Chaperone` keyword or
  a chaperone/factor name still overrides them.
* **Ambiguous symbols**: a gene synonym is accepted when one UniProt entry
  carries it (Slug -> SNAI2) and, when several share it, only if the abstract
  corroborates exactly one (HBA -> haemoglobin alpha, not SCN2A). A primary
  gene or recommended name always beats a synonym, and the organism an
  abstract names beats a human synonym (Rim4 in a yeast paper). Size labels
  that the proteostasis literature reads one way (p62, p97, p38, PHF6) are
  searched as that gene; others (p25, p50) need corroboration.
* **Roles from any paper**: each abstract votes; a protein is an aggregator when
  any paper supports self-aggregation, and every modifier subtype some paper
  supports is kept in `modifier_roles`, so one protein can be both (gelsolin).
  A paper that applies a chaperone noun to the protein votes chaperone whatever
  the model says, because the model entails self-aggregation for almost any
  protein in an amyloid abstract. Proteins with a UniProt `Chaperone` keyword
  are not re-scored, so an aggregator role of such a protein (alphaB-crystallin
  fibrils) is not inferred.
* **Relevance**: in re-validation, a protein whose abstracts never use an
  aggregation term is a `condensate` candidate if they use condensate terms,
  `off_topic` otherwise; an aggregator must be named in a sentence about
  aggregation, or is demoted to its modifier role if a paper supports one.
  NLI abstention keeps the protein. Condensate vocabulary covers stress
  granules, P-bodies, the nucleolus, paraspeckles, Cajal and PML bodies, germ
  granules and LLPS; a condensate term beside any protein sets
  `condensate_evidence`.
* **Where the role is decided**: while mining, only deterministic evidence sets
  a modifier -- a UniProt `Chaperone` keyword, a chaperone-class noun, or a
  dominant disaggregation claim. The broad claims below are scored and stored
  in `role_scores`, never acted on per paper: on the 12873-paper corpus,
  deciding on them with fixed thresholds made 1220 of 1870 proteins modifiers
  and suppressed `forms_aggregates` for 1202, amyloid-beta among them. The role
  needs every abstract of a protein and thresholds calibrated against review,
  so `revalidate_mining.py --rescore-claims` decides it and repairs
  `forms_aggregates` where it does.
* **Role claims**: besides self-aggregation the NLI model scores four
  modifier claims -- chaperone/disaggregase, inhibitor (prevents, reverses or
  degrades another protein's aggregates), regulator (acts on expression,
  processing, modification or clearance of an aggregating protein) and
  associated (binds or co-deposits with another protein's amyloid). The
  leading modifier claim wins when entailed (>= 0.7) and ahead of
  self-aggregation by >= 0.2; the protein is then kept as a modifier with that
  subtype, never exported as an amyloid record.
* **Role**: a UniProt `Chaperone` keyword, a chaperone / heat-shock / DnaJ name,
  or a disaggregase NLI score exceeding the aggregation score by >= 0.2 makes
  the protein a `modifier`, kept in the database but not exported as an amyloid
  record.
* **Status, not deletion**: every rejected protein keeps its row with
  `status = dropped:<reason>`.
* **Triage off by default**: BiomedBERT-Amyloid scored 0.34 % of the 2906
  abstracts of the v2 run >= 0.5 although the query was aggregation-specific, so
  its scores are not used unless `--triage annotate|gate` is given.
* **Validated LLM output**: categorical answers outside the vocabulary, including
  echoes of the option list, count as no answer.
* **Prion policy**: only transmissible mammalian prion pathology is excluded
  (query, prefilter, LLM parsing); prion-like-domain RBPs are kept.

See `running.md` for the commands and `changes_from_source.md` for the measured
defects these guards correct.
