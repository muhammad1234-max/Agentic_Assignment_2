# Agentic AI — Assignment Beta (AI4015)
**FAST National University of Computer and Emerging Sciences, Karachi**  
**Student ID:** `23K-0068` | **Semester:** Fall 2026

---

## Repository Structure

### 📁 [q1/](./q1/) — Empirical Tool Latency Experiment & Latency Budget Analysis
- **Focus:** SaafHawa Agentic Assistant tool-calling latency benchmark across multiple calendar days.
- **[q1/src/](./q1/src/):** Core Python clients and experiment orchestrator (`openaq_client.py`, `open_meteo_client.py`, `heat_index.py`, `knowledge_base.py`, `logger.py`, `analyzer.py`, `report_generator.py`, `experiment.py`).
- **[q1/tests/](./q1/tests/):** Unit test suite covering tool algorithms, data parsing, and statistics.
- **[q1/config/](./q1/config/):** Experiment configuration and coordinates.
- **[q1/data/](./q1/data/):** Offline Pakistani/WHO heatwave advisories and BM25 knowledge base.
- **[q1/results/](./q1/results/):**
  - `raw_tool_latency_log.csv`: Raw empirical timing log (80 valid measurements across two calendar days).
  - `latency_summary.xlsx`: Dynamic 6-sheet Excel workbook with formulas and calculations.
  - `latency_summary.md`: Full formal Markdown report.
  - `latency_summary.json`: Machine-readable summary metrics and validation proof.
  - `charts/latency_by_tool.png`: Comparative visual bar chart.
- **[q1/Assignment_Beta_Worked_Solution.docx](./q1/Assignment_Beta_Worked_Solution.docx):** Worked solution document updated with Q1 empirical latency findings.

---

### 📁 [q2/](./q2/) — Parameter-Efficient Fine-Tuning (PEFT) Benchmark
- **Focus:** Comprehensive evaluation of **Baseline**, **LoRA ($r=8$)**, **QLoRA (NF4 4-bit, $r=8$)**, **DoRA ($r=8$)**, and **AdaLoRA ($\text{init}=12, \text{target}=8$)** with `Qwen/Qwen2.5-1.5B-Instruct` on Glaive function calling (e-commerce customer support tools).
- **[q2/Assignment_Beta_Question_no_2_Student_ID.ipynb](./q2/Assignment_Beta_Question_no_2_Student_ID.ipynb):** Master standalone self-contained Jupyter notebook containing all 27 assignment sections, embedded code engines, and auto-detecting environment setup.
- **[q2/Assignment_Beta_Worked_Solution.docx](./q2/Assignment_Beta_Worked_Solution.docx):** Official submission document with Table 14 (Empirical PEFT Benchmark), Table 16 (Real Failure Cases), and all reflection responses.
- **[q2/src/](./q2/src/):** Complete modular Python pipeline:
  - `environment_check.py`, `dataset_download.py`, `dataset_preprocessing.py`, `dataset_validation.py`
  - `model_loader.py`, `train_baseline.py`, `train_lora.py`, `train_qlora.py`, `train_dora.py`, `train_adalora.py`, `train_rank_sweep.py`
  - `inference.py`, `metrics.py`, `evaluation.py`, `failure_analysis.py`, `generate_report.py`
- **[q2/configs/](./q2/configs/):** `experiment_config.yaml` and `tool_schemas.json` (4 customer support tools: `check_order_status`, `initiate_refund`, `track_shipment`, `escalate_to_human`, plus `no_tool`).
- **[q2/results/](./q2/results/):**
  - `metrics/evaluation_summary.json` & `comparative_metrics.csv`: Full comparative metrics across all 5 PEFT methods.
  - `metrics/rank_sweep_metrics.csv`: Rank ablation study across $r \in \{4, 8, 16, 32\}$.
  - `failures/failure_cases.json`: 5 real failure cases with root cause analyses.
  - `plots/`: Publication-quality comparison charts (`comparative_accuracy_plot.png`, `vram_vs_accuracy_plot.png`, `rank_sweep_plot.png`, `training_time_plot.png`).
- **Cloud & Deployment Links:**
  - **Live Interactive Demo (Hugging Face Space):** [muhammadabbaas12345/peft-q2-experiment](https://huggingface.co/spaces/muhammadabbaas12345/peft-q2-experiment)
  - **Hugging Face Dataset:** [muhammadabbaas12345/peft-q2-customer-support](https://huggingface.co/datasets/muhammadabbaas12345/peft-q2-customer-support)
  - **Execution Guides:** Included for Google Colab, Kaggle GPU, and Lightning AI.
