# DS_Capstone

Capstone projects for the Flatiron School **DS11 Data Science** program by **Eric Bowers**.
Each project lives in its own folder with its own notebook(s), dependencies, and (where
present) a detailed README.

| # | Project | Type | Core tools | Status |
|---|---|---|---|---|
| 1 | [US Traffic Accidents](US%20Traffic%20Accidents/) | Exploratory & business analysis + Tableau dashboard | pandas, matplotlib, seaborn, Tableau | Complete |
| 2 | [Predicting Loan Default](Predicting%20Loan%20Default/) | Supervised ML — binary classification | scikit-learn, pandas | Complete |
| 3 | [Dialogue Summarization](Project%203%20Dialogue%20Summarization/) | NLP — abstractive summarization (BERT encoder + GPT-2 decoder) | PyTorch, Hugging Face Transformers | In progress |

---

## 1 · US Traffic Accidents — Exploratory & Business Analysis

**Scenario:** a data analyst at a Department of Transportation needs to understand when,
where, and under what conditions U.S. traffic accidents happen, to target road-safety
spending.

- **Data:** US Accidents (2016 – March 2023), ~7 million accident records across 49 states
  (`US_Accidents_March23.csv`, from Kaggle; not included in this repo because of its size).
- **Approach:** CRISP-DM business and data understanding, cleaning (datetime parsing,
  imputation, outlier capping, derived time and weather features), then analysis of
  timing, geography, severity drivers, weather, and road features.
- **Key findings:**
  - Accidents peak in the weekday commute windows (7–8 AM and 4–6 PM); weekday volume is
    roughly double weekend volume.
  - Controlled road features (signals, crossings, stop signs) show *lower* average
    severity — most likely a road-type and speed confound.
  - Rain and snow/ice carry a higher share of severe accidents; fog/haze does not.
- **Deliverables:** three recommendations with tracking metrics, condensed CSV exports for
  Tableau, and an interactive Tableau dashboard.

```
US Traffic Accidents/
├── DS_Capstone_Notebook.ipynb   ← full analysis and Tableau export code
├── Data/                        ← condensed CSVs that feed the Tableau dashboard
└── images/
```

## 2 · Predicting Loan Default Across the Credit Funnel

**Scenario:** a consumer-lending platform (LendingClub, 2007–2018) wants to cut credit
losses by scoring default risk at the moment of application.

- **Data:** LendingClub accepted loans (2.26M × 151) and rejected applications
  (27.6M × 9) from Kaggle; raw files are not committed.
- **Approach:** leakage audit that removes 38 post-funding columns; a reproducible
  scikit-learn pipeline with custom transformers; Logistic Regression, Random Forest, and
  gradient boosting compared with cross-validation; stratified and time-based test sets.
- **Results:** tuned HistGradientBoosting reaches **test ROC-AUC 0.731** (0.723 on
  2017–2018 loans), meeting the ≥ 0.70 target without LendingClub's own grades. Approving
  the safest 55% of applicants cuts the approved-book default rate from ~20% to 10.3%.
- **Secondary analysis:** a descriptive model of historical approval decisions (approved vs.
  declined) with a geography-only fairness read.

Full write-up, data instructions, and results: [`Predicting Loan Default/README.md`](Predicting%20Loan%20Default/README.md).

## 3 · Dialogue Summarization with a BERT-based Encoder-Decoder

**Scenario:** a messaging company wants one- or two-sentence summaries so users can catch up
on busy group chats in seconds.

- **Data:** SAMSum, ~16k messenger-style dialogues with human-written summaries.
- **Approach:** a warm-started Hugging Face `EncoderDecoderModel` (BERT encoder, GPT-2
  decoder) fine-tuned on SAMSum, compared against a Lead-3 baseline using ROUGE with
  bootstrap confidence intervals, plus factual-consistency checks. Includes a command-line
  summarizer (`inference.py`).
- **Status:** in progress — results are written into the project README by the notebook's
  final cell after a full GPU run (Google Colab recommended).

Details and run instructions: [`Project 3 Dialogue Summarization/README.md`](Project%203%20Dialogue%20Summarization/README.md).

---

## Running the projects

Each project is independent. From inside a project folder:

```bash
pip install -r requirements.txt   # Projects 2 and 3 include one
jupyter lab
```

Project 1 needs only pandas, NumPy, matplotlib, and seaborn. Large raw datasets are not
stored in this repository; each project explains where to download its data.

## License

Code in this repository is released under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE). The datasets belong to their original publishers and remain under
their own terms (US Accidents and LendingClub via Kaggle; SAMSum via its authors).

## Contact

Eric Bowers · GitHub [@bowers261](https://github.com/bowers261)
