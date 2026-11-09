# Agentic AI — Assignment Beta (AI4015)
**FAST National University of Computer and Emerging Sciences, Karachi**  
**Student ID:** 23K-0068

## Repository Structure

- [q1/](./q1/): **Question 1 — Empirical Tool Latency Experiment & Latency Budget Analysis**
  - [q1/src/](./q1/src/): Core Python clients and experiment orchestrator (openaq_client.py, open_meteo_client.py, heat_index.py, knowledge_base.py, logger.py, nalyzer.py, 
eport_generator.py, experiment.py).
  - [q1/tests/](./q1/tests/): Comprehensive unit test suite covering all tool algorithms, data parsing, and statistics.
  - [q1/config/](./q1/config/): Experiment configuration and coordinates.
  - [q1/data/](./q1/data/): Offline Pakistani/WHO heatwave advisories and BM25 knowledge base.
  - [q1/results/](./q1/results/):
    - 
aw_tool_latency_log.csv: 121-row raw empirical timing log (80 valid measurements across two calendar days).
    - latency_summary.xlsx: 6-sheet dynamic Excel workbook with calculations and formulas.
    - latency_summary.md: Full formal Markdown report.
    - latency_summary.json: Machine-readable summary metrics and validation proof.
    - charts/latency_by_tool.png: Comparative visual bar chart.
  - [q1/Assignment_Beta_Worked_Solution.docx](./q1/Assignment_Beta_Worked_Solution.docx): Completed assignment worked solution document updated with empirical latency findings.
