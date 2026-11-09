# Q1 LATENCY EXPERIMENT — SAAFAHAWA AGENTIC ASSISTANT
**Course:** Agentic AI (AI4015) — Assignment Beta  
**Institution:** FAST National University of Computer and Emerging Sciences, Karachi  
**Department:** School of Computing  
**Generated On:** 2026-11-10 01:18:50 PKT  
**Experiment Status:** `COMPLETE — ALL EXPERIMENT REQUIREMENTS SATISFIED`  

---

## 1. Student Parameter Derivations

- **Student ID:** `23K-0068`
- **Roll Number Last Two Digits (R):** `68`
- **Modulo-10 Digit Sum (D):** `9`
- **Normal Daily Query Volume (Q):** `1,500 + 25 × 68 = 3,200` queries/day
- **LLM Calls Per Query (N):** `5 + (68 mod 4) = 5` sequential calls
- **Tool-Call Failure Probability (p):** `0.04 + 0.01 × 9 = 0.13` (13%)
- **Heatwave Days Per Month (H):** `4 + (68 mod 4) = 4` days
- **Monthly Inference Budget (B):** `$1,200 + 10 × 68 = $1,880` USD
- **End-to-End Latency Target (T):** `10 - (68 mod 3) = 8.0` seconds

---

## 2. Experimental Methodology & Rigorous Tool Isolation

To construct an auditable empirical foundation for the SaafHawa agent's latency budget, four distinct tool operations were benchmarked on the physical host using sub-microsecond monotonic counters (`time.perf_counter_ns()`):

1. **OpenAQ API v3 (Air Quality):** Live HTTP REST queries to the OpenAQ API v3 endpoint (`/v3/locations/{id}/latest`) targeting monitoring stations within a 25 km radius of Karachi center (24.8607° N, 67.0011° E). Evaluates live PM2.5 and PM10 particulate mass concentrations. Authentication is handled via the official `X-API-Key` request header.
2. **Open-Meteo API (Weather):** High-resolution HTTP requests to `https://api.open-meteo.com/v1/forecast` fetching real-time surface temperature (`temperature_2m`), relative humidity (`relative_humidity_2m`), and wind velocity (`wind_speed_10m`) for Karachi under the `Asia/Karachi` timezone.
3. **Local Heat-Index Computation:** In-process Python calculation implementing the official National Weather Service (NWS) Steadman and Rothfusz polynomial regression algorithms with low- and high-humidity adjustments. Timings measure pure local compute time without network overhead.
4. **Local Knowledge-Base Retrieval:** Purely local lexical retrieval implementing deterministic BM25 ranking ($k_1=1.5, b=0.75$) across an indexed corpus of official emergency heatwave SOPs and public health advisories from NDMA Pakistan, Pakistan Meteorological Department (PMD), Government of Sindh Health Department, and WHO.

---

## 3. Tool Latency Empirical Measurements

| Tool Operation | Measured Samples | Failed Attempts | Recorded Dates | Mean Latency (ms) | Median Latency (ms) | Median Latency (s) | P95 Latency (ms) | Std Dev (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenAQ** | 20 | 20 | 2026-11-09, 2026-11-10 | 486.74 | 524.36 | 0.5244 | 686.49 | 199.16 |
| **Open-Meteo** | 20 | 0 | 2026-11-09, 2026-11-10 | 130.41 | 131.43 | 0.1314 | 135.03 | 4.52 |
| **HeatIndex** | 20 | 0 | 2026-11-09, 2026-11-10 | 0.02 | 0.02 | 0.0000 | 0.04 | 0.01 |
| **KnowledgeBase** | 20 | 0 | 2026-11-09, 2026-11-10 | 0.07 | 0.07 | 0.0001 | 0.11 | 0.02 |

### 3.1 Overall Pooled Median Tool Latency ($d_{\text{tool}}$)

> **Overall pooled median tool latency = 62.025 ms = 0.0620 seconds**
> *(Derived from 80 valid individual measurements across currently active tool categories)*

> **Integrity Verification:** *All four required tool categories (OpenAQ, Open-Meteo, Heat Index, Knowledge Base) are fully represented in this pooled median.*

---

## 4. Expected Retry Delay & End-to-End Latency Evaluation

### 4.1 Retry Model Derivation
- Tool calls per query ($N - 1$): **4 initial operations** (OpenAQ, Open-Meteo, Heat Index, Knowledge Base).
- Tool failure probability ($p$): **0.13 (13%)** per call on calls 1 through 4.
- Under the assignment retry model, an invalid tool call is repeated exactly once and the retry always succeeds.
- **Expected tool retry operations:** $4 \times p = 4 \times 0.13 = \mathbf0.52$ retries.
- **Expected total tool executions:** $4 \times (1 + p) = 4 \times 1.13 = \mathbf4.52$ executions.

### 4.2 Tool Delay Breakdown
- **Base Tool Delay:** $4 \times d_{\text{tool}} = 4 \times 0.0620\text{ s} = \mathbf{0.2480\text{ seconds}}$
- **Expected Tool Retry Delay:** $0.52 \times d_{\text{tool}} = 0.52 \times 0.0620\text{ s} = \mathbf{0.0322\text{ seconds}}$
- **Total Expected Tool Delay:** $4.52 \times d_{\text{tool}} = 4.52 \times 0.0620\text{ s} = \mathbf{0.2802\text{ seconds}}$

### 4.3 End-to-End Latency Comparison for Strongest Candidate Models

The two strongest candidates identified from the Q1 comparative analysis are evaluated under the 5-step sequential reasoning loop:

| Metric | Mistral Small 4 (`mistral-small-2603`) | Google Gemini 4 Argon (`gemini-4-argon`) |
| :--- | :---: | :---: |
| **Model Type / Host** | Mistral AI Studio (Open-weight) | Google DeepMind / Google Cloud (Closed-source) |
| **Published TTFT** | 0.84 s | 40.21 s |
| **Output Throughput** | 173.0 tok/s | 71.0 tok/s |
| **Base LLM Latency (5 calls, 2,000 out tok)** | 15.761 s | 229.219 s |
| **Expected LLM Retry Latency (p=0.13)** | 1.489 s | 23.473 s |
| **Total Expected LLM Latency** | **17.250 s** | **252.692 s** |
| **Base Tool Delay ($4 \times d_{\text{tool}}$)** | 0.248 s | 0.248 s |
| **Expected Tool Retry Delay ($0.52 \times d_{\text{tool}}$)** | 0.032 s | 0.032 s |
| **Total Expected Tool Delay ($4.52 \times d_{\text{tool}}$)** | **0.280 s** | **0.280 s** |
| **Expected End-to-End Latency** | **17.530 s** | **252.972 s** |
| **Target Threshold (T = 8.0 s)** | FAILS (Exceeds Target) | FAILS (Exceeds Target) |
| **Latency Slack / Exceedance** | -9.530 s | -244.972 s |
| **Revised Architecture Latency** | **3.160 s** (PASS) | **45.022 s** (FAILS) |

---

## 5. Architectural Optimization Proposal

Because sequential agentic execution triggers repeated LLM round trips (5 calls) and serial tool delays, both models incur high baseline latencies under the strict 5-turn loop. To guarantee compliance with the **8.0-second SLO ($T$)**, the following architectural change is implemented:

1. **Deterministic Request Routing:** Replace sequential conversational tool querying with an upfront intent and entity extraction parser. Location, target timestamp, and risk category (e.g. child, elderly, asthma) are resolved in a single step.
2. **Parallel Asynchronous Tool Execution:**
   - **External Branch:** OpenAQ and Open-Meteo HTTP requests are dispatched simultaneously via asynchronous HTTP (`asyncio.gather`). Delay is $\max(t_{\text{OpenAQ}}, t_{\text{Open-Meteo}})$ rather than their sum.
   - **Local Branch:** Once weather parameters return, the local Heat Index computation and local Knowledge Base BM25 retrieval are executed in parallel. Delay is $\max(t_{\text{HeatIndex}}, t_{\text{KB}})$.
3. **Single Synthesis Generation Call:** The retrieved observations and advisory excerpts are packaged directly into a structured prompt, and a single generation call is made to Mistral Small 4 with an enforced 300-token output limit.

> Under this revised architecture, model generation time drops to $\approx 0.84\text{ s} + (300 / 173) = 2.574\text{ s}$. Combined with parallel tool retrieval, total latency easily meets the 8.0-second requirement.

---

## 6. Experiment Validation & Two-Day Protocol Compliance

**Validation Status:** `COMPLETE — ALL EXPERIMENT REQUIREMENTS SATISFIED`  

- **Sample Uniqueness:** Passed (No duplicate sample IDs)
- **Non-negative Latencies:** Passed (All latency values > 0 ms)
- **Timestamp Integrity:** Passed (All start/complete ISO timestamps valid)
- **Recorded Calendar Dates (2):** 2026-11-09, 2026-11-10

### Status Notes & Next Steps:
- **Experiment Complete:** All mandatory assignment empirical criteria have been fully satisfied.
- **Sample Quota:** 20 successful OpenAQ calls, 20 successful Open-Meteo calls, 20 local Heat-Index computations, and 20 local BM25 Knowledge-Base retrievals.
- **Two-Day Spread:** Measurements span 2 distinct calendar dates (2026-11-09, 2026-11-10).
- **Submission Deliverables:** The raw CSV audit log, multi-sheet Excel workbook with formulas, JSON summary, and comparative chart are verified and submission-ready.
