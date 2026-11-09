# Q1 Latency Experiment — SaafHawa Agentic Assistant

**Course:** Agentic AI (AI4015) — Assignment Beta  
**Student ID:** `23K-0068`  
**Campus / Department:** FAST-NUCES Karachi / School of Computing  
**Semester:** Fall 2026  

---

## 1. Executive Summary & Purpose

This project implements, executes, measures, and documents the empirical latency benchmark required by **Question 1 (Section 2, Item 6)** of Assignment Beta.

The SaafHawa agent is an AI-powered conversational assistant for Karachi citizens seeking advice on extreme heatwaves and air pollution. Answering a citizen query requires four sequential tool operations before generating a final response:
1. **Air Quality Tool:** Queries OpenAQ API v3 for PM2.5 and PM10 particulate levels.
2. **Weather Tool:** Queries Open-Meteo for ambient temperature, relative humidity, and wind speed.
3. **Heat-Index Tool:** Computes the National Weather Service (NWS) Heat Index locally from weather parameters.
4. **Knowledge-Base Retrieval:** Queries a local repository of official emergency heatwave SOPs and advisories (NDMA, PMD, Sindh Health Department, WHO) using deterministic BM25 lexical ranking.

The assignment mandates measuring each operation at least 20 times across at least **two different calendar days** on the student's actual physical machine, calculating the **pooled median latency ($d_{\text{tool}}$)**, incorporating tool and LLM retries ($p = 0.13$), and determining whether the system satisfies the **$T = 8.0$ second** end-to-end latency budget.

---

## 2. Derived Assignment Parameters (Student ID: 23K-0068)

From the roll number `23K-0068`:
- Numeric digits: $2, 3, 0, 0, 6, 8$
- Last two digits: $R = 68$
- Modulo-10 digit sum: $D = (2 + 3 + 0 + 0 + 6 + 8) \pmod{10} = 19 \pmod{10} = 9$

| Parameter | Formula | Value |
| :--- | :--- | :--- |
| Normal daily queries ($Q$) | $1,500 + 25 \times R$ | **3,200** queries/day |
| LLM calls per query ($N$) | $5 + (R \bmod 4)$ | **5** calls |
| Tool-call failure probability ($p$) | $0.04 + 0.01 \times D$ | **0.13** (13%) |
| Heatwave days per month ($H$) | $4 + (R \bmod 4)$ | **4** days |
| Monthly inference budget ($B$) | $\$1,200 + 10 \times R$ | **$1,880** USD |
| End-to-end latency target ($T$) | $10 - (R \bmod 3)$ | **8.0** seconds |

---

## 3. Project Architecture

```text
Q1_Latency_Experiment/
│
├── README.md                           <- Comprehensive experimental manual
├── requirements.txt                    <- Python dependencies
├── .env                                <- Local environment variables (API keys)
├── .env.example                        <- Template for API keys
├── .gitignore                          <- Excludes virtualenv and secrets
│
├── config/
│   └── experiment_config.json          <- Student parameters, coordinates, endpoints
│
├── data/
│   ├── knowledge_base/                 <- Official Karachi/Sindh advisory corpus
│   │   ├── ndma_heatwave_guidelines_pakistan.txt
│   │   ├── pmd_heatwave_classification_advisory.txt
│   │   ├── sindh_health_dept_heatstroke_sop.txt
│   │   ├── who_heatwaves_public_health_factsheet.txt
│   │   ├── sindh_epa_urban_air_quality_advisory.txt
│   │   ├── sources_metadata.json       <- Citations & URLs
│   │   └── test_queries.json           <- 20 documented bilingual test queries
│   └── raw_responses/                  <- Raw network audit dumps
│
├── src/
│   ├── __init__.py
│   ├── experiment.py                   <- Main automated two-day orchestrator
│   ├── openaq_client.py                <- OpenAQ API v3 client with location discovery
│   ├── open_meteo_client.py            <- Open-Meteo weather client
│   ├── heat_index.py                   <- NWS Rothfusz / Steadman heat-index calculator
│   ├── knowledge_base.py               <- Deterministic BM25 local lexical retriever
│   ├── logger.py                       <- Append-only raw CSV logger
│   ├── analyzer.py                     <- Statistical analyzer, pooled median, retry budget
│   └── report_generator.py             <- Generates JSON, Markdown, Excel, and chart
│
├── tests/
│   ├── test_heat_index.py              <- Boundary and regression tests for NWS formula
│   ├── test_api_parsing.py             <- Fixture-based network and schema tests
│   ├── test_logger.py                  <- Append-only integrity and count verification
│   └── test_analyzer.py                <- Pooled median, date validation, and budget tests
│
└── results/
    ├── raw_tool_latency_log.csv        <- Append-only raw timing log
    ├── experiment_metadata.json        <- Selected OpenAQ station metadata
    ├── latency_summary.json            <- Full machine-readable results
    ├── latency_summary.md              <- University submission academic report
    ├── latency_summary.xlsx            <- Multi-sheet formatted Excel workbook
    └── charts/
        └── latency_by_tool.png         <- Mean vs median comparison chart
```

---

## 4. Setup & API Key Configuration

### 4.1 Virtual Environment Activation
The project is configured with a dedicated local virtual environment inside `Q1_Latency_Experiment/.venv`:

```powershell
# From the project root (c:\Users\hp z book\Desktop\q1 experiment\Q1_Latency_Experiment):
.venv\Scripts\Activate.ps1
```

### 4.2 OpenAQ API v3 Key Setup
The OpenAQ API v3 requires an API key in the `X-API-Key` HTTP header.

1. Register for a free API key at [OpenAQ](https://explore.openaq.org/) or [OpenAQ API Portal](https://openaq.org/).
2. Open the `.env` file in `Q1_Latency_Experiment/`:
   ```env
   OPENAQ_API_KEY=your_actual_openaq_key_here
   ```
3. Save the file. Never commit `.env` to version control (it is excluded via `.gitignore`).

---

## 5. Execution Workflow

### 5.1 Running the Suite
Execute the main experiment runner:

```powershell
python -m src.experiment
```

The runner automatically:
1. Loads `.env` credentials and `config/experiment_config.json`.
2. Checks the local calendar date in `Asia/Karachi`.
3. Checks existing measurements in `results/raw_tool_latency_log.csv`.
4. Runs warm-up iterations (clearly tagged and excluded from statistics).
5. Gathers up to the daily target of 10 successful samples per tool.
6. Automatically pauses between API requests (1.5 seconds) to respect server rate limits.
7. Logs all attempts, latencies, HTTP statuses, and inputs/outputs to `raw_tool_latency_log.csv`.
8. Updates the statistical summary, markdown report, Excel workbook, and chart.

### 5.2 Two-Calendar-Day Protocol
- **Day 1 Session:** Run `python -m src.experiment`. 10 samples per tool are recorded. Status is marked `PARTIAL — SECOND EXPERIMENT DAY REQUIRED` (or `PARTIAL — INSUFFICIENT SAMPLES` if an API key is missing).
- **Day 2 Session:** On a subsequent calendar day (after 00:00 Asia/Karachi time), run `python -m src.experiment` again. The runner detects the new calendar date, records the remaining 10 samples per tool without overwriting previous data, validates that measurements span at least two distinct calendar days, and updates the status to `COMPLETE — ALL EXPERIMENT REQUIREMENTS SATISFIED`.

---

## 6. Running Unit Tests

To run the full unit test suite (19 tests):

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tests verify:
- Accurate NWS Rothfusz regression and low/high humidity adjustments.
- Open-Meteo and OpenAQ response parsing under success and error modes.
- Append-only preservation of raw CSV rows.
- Absolute rejection of single-day runs as a complete two-day experiment.
- Calculation of the true pooled median directly from all valid measurements.

---

## 7. Measured vs. Published Benchmark Disclosures

In strict adherence to experimental integrity:
- **Measured on host machine:**
  - Network round-trip and JSON parsing latency for Open-Meteo.
  - Network round-trip and JSON parsing latency for OpenAQ (when API key is active).
  - High-resolution in-process CPU execution time for NWS Heat Index calculation.
  - High-resolution in-process CPU execution time for BM25 Knowledge-Base retrieval.
  - Overall pooled median latency ($d_{\text{tool}}$).
- **Published Benchmark Citations (Not Measured Locally):**
  - **Mistral Small 4:** TTFT $0.84\text{ s}$, throughput $173\text{ tok/s}$ (Mistral AI Studio / Artificial Analysis snapshot).
  - **Google Gemini 4 Argon:** TTFT $40.21\text{ s}$, throughput $71\text{ tok/s}$ (Google Cloud model evaluation card / Artificial Analysis snapshot).

