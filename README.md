# VeritasLens

**Real-Time Hallucination Detection powered by Gemma 4**

[![Gemma 4 Good Hackathon](https://img.shields.io/badge/Gemma%204%20Good-Hackathon%202026-orange)](https://www.kaggle.com/competitions/gemma-4-good)
[![Unsloth](https://img.shields.io/badge/Fine--tuned%20with-Unsloth-blue)](https://github.com/unslothai/unsloth)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> Built for the Gemma 4 Good Hackathon | Safety & Trust Track | Main Track | Unsloth Special
> Author: Jacob O. | UCSD Physics | Incoming Berkeley MIDS 2026

---

## What it does

VeritasLens takes any AI generated text, splits it into individual claims, retrieves structured evidence for each one, and returns a verdict with cited sources and corrected text. It does not produce a binary hallucination flag. Every verdict is earned, evidenced, and explained.

**Five verdict types:**

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Claim is accurate per retrieved evidence |
| `UNSUPPORTED` | Claim is false - evidence directly contradicts it |
| `CONTESTED` | Sources genuinely disagree |
| `UNVERIFIABLE` | Cannot determine with available evidence |
| `OPINION` | Subjective statement, not verifiable |

Every `UNSUPPORTED` verdict includes a corrected version of the claim.

---

## Demo output

```
Input:
  "The Great Wall of China is visible from space.
   Einstein failed math as a child.
   Marie Curie won three Nobel Prizes."

Output:
  Overall reliability: 0%  |  Claims: 3  |  Evidence calls: 6

  [!!] Claim 1 -- 95% confidence
  Not supported -- likely hallucination
  "The Great Wall of China is visible from space."
  The evidence explicitly states the Great Wall is NOT visible from
  space with the naked eye.
  Sources: NASA; China National Space Administration
  Corrected: The Great Wall of China is not visible from space with
  the naked eye.

  [!!] Claim 2 -- 98% confidence
  Not supported -- likely hallucination
  "Einstein failed math as a child."
  The evidence contradicts the claim -- Einstein excelled at
  mathematics from an early age.
  Sources: ETH Zurich historical records; Einstein Archive
  Corrected: Einstein did not fail math as a child.

  [!!] Claim 3 -- 97% confidence
  Not supported -- likely hallucination
  "Marie Curie won three Nobel Prizes."
  The evidence clearly states Marie Curie won exactly two Nobel Prizes.
  Sources: Nobel Prize Foundation official records
  Corrected: Marie Curie won two Nobel Prizes.
```
<img width="1856" height="2901" alt="image" src="https://github.com/user-attachments/assets/052a921d-ee39-4717-be83-b3562736a460" />

---

## Architecture

```
Input text
    |
    v
Sentence splitter
    |
    v
Gemma 4 E4B (Unsloth QLoRA fine-tuned)
    |
    +--> fetch_evidence(claim, search_query)
    |        |--> Tier 1: Offline KB  (12 entries, instant, no network)
    |        |--> Tier 2: Wikipedia REST API  (free, no key required)
    |        +--> Tier 3: Brave Search  (optional, set BRAVE_API_KEY)
    |
    +--> check_knowledge_base(claim, domain)
    |
    +--> resolve_entity(entity_name, entity_type, claimed_attribute)
    |
    v
Multi-round tool loop with regression guard
    |
    v
Reliability recomputed from verdict distribution (never trusted from model)
    |
    v
GroundingResult + Gradio demo
```

**Key design decisions:**

- **Reliability computed from verdicts, not model output.** The model's self-reported score is ignored. Reliability is derived deterministically: SUPPORTED=1.0, UNSUPPORTED=0.0, CONTESTED=0.5. Three claims all labeled UNSUPPORTED = 0% reliability. This was a real bug in an earlier version - the model was reporting 95% reliability for entirely false content.
- **Three-tier evidence fallback.** Offline KB is instant and requires no network. Wikipedia REST covers the long tail at no cost. Brave Search handles current events when a key is available.
- **Regression guard.** After QLoRA fine-tuning, the model can skip tool calls and output JSON directly. The guard detects all-UNVERIFIABLE or tool-free outputs and injects a structured nudge with a concrete tool_call format example before retrying.
- **Causal mediation interpretability.** Token attribution measures the shift in the model's true/false logit difference when each token is replaced with UNK. This runs as a separate probe on the base model - it surfaces internal sensitivity, not a trace of the grounding pipeline's verdict path.

---

## Evaluation

10-case held-out suite (no overlap with training data):

```
 Accuracy: 9/10 = 90%

 [PASS] geography     UNSUPPORTED  correct
 [PASS] biography     UNSUPPORTED  correct  (kb fired)
 [PASS] history       UNSUPPORTED  correct  (kb fired)
 [PASS] medicine      UNSUPPORTED  correct  (kb fired)
 [PASS] neuroscience  UNSUPPORTED  correct
 [PASS] quotes        UNSUPPORTED  correct  (kb fired)
 [PASS] science       SUPPORTED    correct
 [PASS] geography     SUPPORTED    correct
 [FAIL] astronomy     UNSUPPORTED  got UNVERIFIABLE  (KB gap, honest)
 [PASS] history       SUPPORTED    correct
```

The one failure (Moon/Ganymede) is documented honestly: the KB does not contain a Moon size entry, Wikipedia returned an ambiguous snippet, and UNVERIFIABLE is the correct behavior under insufficient evidence. The system did not fabricate a verdict.

---

## Model and Training

| | |
|---|---|
| Base model | `unsloth/gemma-4-E4B-it` (~5.98B params, 4-bit) |
| Fine-tuning | Unsloth QLoRA, r=8, alpha=8 |
| Trainable params | 18,350,080 / 5,997,636,128 (0.31%) |
| Training examples | 208 (8 seed + 200 TruthfulQA) |
| Training time | 4.7 min on T4 GPU |
| Loss curve | 13.97 → 0.61 |
| Export | GGUF q4_k_m via llama.cpp |

`train_on_responses_only` is used so gradients only flow through verdict outputs, not repeated tool call formats. Seed examples use 1-3 tool calls depending on claim complexity.

---

## Quickstart

**Run the notebook:** Open `veritaslens.ipynb` in Google Colab with a T4 GPU and run all cells top to bottom. No API keys required for the core pipeline.

**Local inference via Ollama:**
```bash
ollama create veritaslens -f ./veritaslens-gguf_gguf/Modelfile
ollama run veritaslens
```

**Python API:**
```python
result = ground_text("Marie Curie won three Nobel Prizes.")

print(result.overall_reliability)    # 0.0
print(result.claims[0]["verdict"])   # UNSUPPORTED
print(result.claims[0]["corrected"]) # Marie Curie won two Nobel Prizes.
print(result.tools_called)           # list of evidence calls that fired
```

---

## Honest Limitations

- **KB coverage:** 12 curated entries. Claims outside the KB depend on Wikipedia quality and disambiguation.
- **Latency:** 1-3 min per analysis on T4 GPU. GGUF reduces this for local use.
- **Sentence splitting:** Complex compound sentences may not split cleanly on punctuation boundaries.
- **Comparative superlatives:** Claims like "largest X in Y" require the KB to contain the correct comparator explicitly. Wikipedia summaries rarely include such rankings.
- **TruthfulQA training skew:** All 200 TruthfulQA training examples use verified correct answers (SUPPORTED). No UNSUPPORTED TruthfulQA examples are in training - this may cause slight SUPPORTED bias on out-of-KB claims.
- **Interpretability scope:** The causal mediation probe runs on the base model's binary classification signal, not the grounded verdict pathway.
- **Eval suite size:** 10 cases is a proof-of-concept. Production use would require a larger, more diverse benchmark.

---

## Repository Structure

```
veritaslens.ipynb          Main notebook (all cells, end-to-end)
README.md                  This file
```

---

## License

CC BY 4.0. Model weights derived from Gemma 4 E4B, subject to the Gemma Terms of Use.

---

## Citation

```
@misc{veritaslens2026,
  author    = {Jacob Ortiz},
  title     = {VeritasLens: Real-Time Hallucination Detection with Gemma 4},
  year      = {2026},
  note      = {Gemma 4 Good Hackathon, Google DeepMind x Kaggle},
  url       = {https://github.com/agentjakey/veritaslens}
}
```
