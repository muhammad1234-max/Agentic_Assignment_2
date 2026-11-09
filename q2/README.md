# Assignment Beta — Question 2: Parameter-Efficient Fine-Tuning
## LoRA vs QLoRA vs DoRA vs AdaLoRA

**Student:** 23K-0068 | **University:** FAST-NUCES Karachi | **Course:** AI4015 Agentic AI | **Semester:** Fall 2026

---

## Quick Start (Google Colab)

### Step 1: Upload to Google Drive
1. Download this entire `Assignment_Beta_Question_2/` folder
2. Upload it to your Google Drive at: `My Drive/Assignment_Beta_Question_2/`

### Step 2: Open the Notebook in Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Open Notebook → Google Drive**
3. Navigate to `Assignment_Beta_Question_2/Assignment_Beta_Question_no_2_Student_ID.ipynb`
4. Open it

### Step 3: Select GPU Runtime
1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** (free tier) or **A100** (if you have Colab Pro)
3. Click **Save**

### Step 4: Run the Experiment
1. Click **Runtime → Run all** (or Ctrl+F9)
2. The notebook will:
   - Mount Google Drive automatically
   - Install all dependencies
   - Download model and dataset
   - Run all 5 experiments + rank sweep
   - Generate all plots and reports
   - Save everything to Google Drive

### Checkpointing & Resume
- **If Colab disconnects:** Simply reopen the notebook and click "Run All" again. 
  The checkpoint system will skip all completed experiments and resume where it left off.
- **Progress tracking:** Each experiment prints clear stage headers, progress bars, 
  and estimated time remaining.
- **All results saved to Drive:** Adapters, predictions, metrics, and plots are saved 
  to Google Drive after each experiment completes.

---

## Project Structure

```
Assignment_Beta_Question_2/
├── Assignment_Beta_Question_no_2_Student_ID.ipynb  ← Main notebook (run in Colab)
├── src/                                             ← Source scripts
│   ├── environment_check.py
│   ├── dataset_download.py
│   ├── dataset_preprocessing.py
│   ├── dataset_validation.py
│   ├── model_loader.py
│   ├── train_baseline.py
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── train_dora.py
│   ├── train_adalora.py
│   ├── train_rank_sweep.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── failure_analysis.py
│   └── generate_report.py
├── data/                   ← Dataset files (created during execution)
├── adapters/               ← Trained adapters (saved during execution)
├── results/                ← Metrics, predictions, plots (saved during execution)
├── reports/                ← Final report (DOCX + PDF)
├── configs/                ← Experiment configuration
│   ├── experiment_config.yaml
│   └── tool_schemas.json
├── requirements.txt
└── README.md               ← This file
```

## Experiment Overview

| Run | Method | Base Precision | Rank |
|-----|--------|---------------|------|
| 0 | Baseline (no fine-tuning) | Original | N/A |
| 1 | LoRA | BF16 | r=8 |
| 2 | QLoRA | NF4 4-bit | r=8 |
| 3 | DoRA | BF16 | r=8 |
| 4 | AdaLoRA | BF16 | init=12, target=8 |

After the 5-way comparison, the best method is trained at r=4, 8, 16, 32.

## Hardware Requirements

- **Minimum:** Google Colab Free (T4 GPU, 15 GB VRAM)
- **Recommended:** Google Colab Pro (A100, 40 GB VRAM) for faster training
- **Estimated time:** ~2-3 hours on T4, ~45 min on A100

## Reproducibility

- Random seed: 42
- All experiments use identical dataset splits, hyperparameters, and evaluation code
- Raw predictions saved for every model and every test example
- Environment details logged automatically

