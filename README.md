# GCI World 2026 September Final Assignment

This project contains a reproducible analysis and a client-facing proposal for Company A, an anonymized wireless telecommunications provider. The recommendation is to run a measured retention pilot, not to deploy the current model at scale: the holdout ROC-AUC is 0.614 and the highest-risk decile captures 13.6% of churners.

## Deliverables

- `output/final_assignment.pdf` - 12-slide business proposal, within the 15-slide limit
- `output/final_assignment.html` - editable, self-contained slide source
- `final_assignment.ipynb` - reproducible analysis and model walkthrough
- `output/analysis_results.json` - aggregate results used in the presentation; no customer-level records
- `src/analysis.py` - reusable analysis, logistic-regression model, and impact scenario calculations
- `references.md` - sources used in the slides and AI disclosure note

To rebuild the PDF from the HTML source on Windows with Microsoft Edge, run `./scripts/build_pdf.ps1`.

## Data setup

The provided course archive contains two customer-level files. Copy them to `data/telecom/`:

- `Client.csv`
- `Record.csv`

The CSVs are excluded from Git because they contain 100,000 linked customer records. They are not needed to review the proposal, but are required to rerun the analysis. The project does not use any external customer data.

## Reproduce the analysis

Use Python 3.10 or newer, then install the packages in `requirements.txt`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the notebook from the repository root or execute the analysis directly:

```bash
python -c "from src.analysis import run_analysis; print(run_analysis('data/telecom', 'output'))"
```

The run writes aggregate model and exploratory analysis results to `output/analysis_results.json`. It checks that each source table has unique customer IDs and that the join is one-to-one. The deterministic ID-hash holdout and fixed model optimization make the score reproducible without splitting data based on the target.

## What the analysis says

The joined sample has 100,000 customers and 100 columns. `churn` is 1 for customers who churned 31-60 days after the observation date, according to the supplied course overview. The sample churn rate is 49.6%; this is not directly comparable with annual industry churn benchmarks.

Customers with equipment aged 730 days or more have a 57.9% churn rate, versus 41.7% for equipment under 180 days. Churners use fewer monthly minutes on average (483 versus 543), while average revenue differs little (58.2 versus 59.2 dataset units).

The model is an L2-regularized logistic regression using 28 numeric service, usage, account, and equipment features. It excludes demographic, credit, and ethnicity fields. On a deterministic 80/20 holdout, ROC-AUC is 0.614. The top-risk decile has 67.1% churn versus 49.4% overall validation churn (1.36x lift) and captures 13.6% of churners. This is modest ranking power and supports a controlled pilot only.

## Business case limits

The impact sensitivity in the slides assumes 100,000 monitored customers, contacts to the top 10%, six retained months, and a contact cost of five units per targeted account. Revenue and contact cost are expressed in the same assumed units because the supplied data dictionary does not specify currency. The scenarios are illustrative, exclude discounts and service costs, and do not represent a forecast. Replace these assumptions with Company A's current margin and offer economics before any investment decision.

## Assignment submission checklist

The course instructions require a PDF, reproducible notebook, and matching list of references. Before uploading, rename the files to the Omnicampus account name if needed. The instructions also require a ChatGPT conversation URL when generative AI is used; add the share URL for this conversation to the Omnicampus references field before submission.

