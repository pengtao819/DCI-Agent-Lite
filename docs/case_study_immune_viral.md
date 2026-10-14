# Case Study: Semantic Search in Action

## Query
> Find documents about how the immune system responds to viral infections.
> Use semantic search if exact keyword search fails.

**Corpus**: BRIGHT-Biology (57,145 documents, 5.7k topics)
**Model**: DeepSeek-Flash
**Date**: 2026-10-14

## Why this case matters

Pure lexical search (`rg`) fails on this query. Exact phrase search returns **0 hits**,
because no document in the corpus contains the literal string
"immune response to viral infection". The agent needs semantic retrieval to bridge
this vocabulary gap.

## Agent behavior observed

The agent demonstrated a **multi-round, tool-alternating** workflow, not a
single-shot RAG-style retrieval:

1. **Environment exploration**: `ls`, `find` to discover `.semantic_index/`
2. **Tool discovery**: grepped project source, found and read `semantic_search.py`
3. **Semantic retrieval**: called the script 6+ times with different query formulations
4. **Lexical cross-validation**: used `grep` to find docs mentioning both "immune" and "virus"
5. **Evidence inspection**: read top-ranked documents in full
6. **Self-recovery**: hit a Windows GBK encoding error, diagnosed it, and
   re-ran with `PYTHONIOENCODING=utf-8`

## Quantitative summary

| Metric | Value |
|---|---|
| Total tool calls | ~14 (12 bash, 2 read) |
| Semantic search invocations | 6 |
| Lexical search invocations | 3 |
| Wall-clock time | 7m 40s |
| Errors encountered | 2 (path typo, encoding) |
| Errors self-recovered | 2 |

## Key evidence of hybrid behavior

- Lexical and semantic tools were **alternated**, not used in a fixed pipeline
- Semantic search was used to **expand recall**, then `grep`/`cat` was used to
  **verify and inspect** the retrieved candidates
- The agent did not stop after a single retrieval; it ran **multiple query
  formulations** and aggregated results

## Top semantic hits (score)

| Score | Document |
|---|---|
| 0.862 | `cancer_not_communicable/PMC7150340_10_0.txt` |
| 0.819 | `cancer_not_communicable/PMC7150340_9_0.txt` |
| 0.789 | `genetic_sequence_of_SARS-CoV-2/Virus_5_17.txt` |
| 0.768 | `cancer_not_communicable/PMC7150340_11_0.txt` |
| 0.761 | `genetic_sequence_of_SARS-CoV-2/Virus_5_22.txt` |

## Full trajectory

See `conversation_full.json` in this directory for the complete tool-call log.