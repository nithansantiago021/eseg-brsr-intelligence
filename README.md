# ESG / BRSR Report Intelligence Platform

## Project layout

```
app.py                                    Streamlit entrypoint — 2 sections
agent/compliance_sentinel.py              The ESG/BRSR Compliance Sentinel agent
agent/history_store.py                    memory for the Compliance Sentinel agent
agent/model_provider.py                   Builds the Strands `Model` the Compliance Sentinel agent runs on
agent/tools.py                            The Compliance Sentinel agent's tools
config/settings.py                        All config, incl. GROQ_API_KEY from env
src/preprocessing/preprocessor.py         PDF structure extraction + markdown-aware chunking
src/embedding/embedder.py                 SentenceTransformer + ephemeral ChromaDB
src/embedding/indexer.py                  Load → chunk → stamp metadata → embed → store
src/retrieval/retriever.py                Two-stage retrieval + metadata filters
src/evaluation/ragas_guard.py             Faithfulness / answer / context relevancy gates
src/rag/rag_pipeline.py                   Prompting, citations, chart-intent hook
src/metrics/schema.py                     ESGMetric structured record
src/metrics/normalizer.py                 Synonym matching + unit compatibility
src/metrics/extractor.py                  Table chunk → ESGMetric records
src/visualization/chart_generator.py      Question → Plotly chart (or refusal)
src/visualization/dashboard.py            Principle/company/year dashboard builder
src/summarization/summarizer.py           Structured, cited ESG summary generation
.env.example
```

## Running locally

```bash
cp .env.example .env   # then fill in GROQ_API_KEY
pip install -r requirements.txt
python -m agent.compliance_sentinel --file report.pdf --company "ABCD Corp" --year FY2024-25
```
or

```bash
streamlit run app.py
```

## Compliance Sentinel Agent (Agents for Humans hackathon)

`agent/` adds an autonomous Strands Agents SDK
agent that reviews a newly filed report end-to-end — indexes it, checks for
missing BRSR disclosures, compares against the company's last filing,
verifies anything ambiguous before concluding, writes a cited memo, and
**only notifies a human when something actually needs attention**. 

## Architecture
![Diagram](./docs/architecture.svg)

## Known limitations

- **Metric extraction is heuristic.** It only converts a table cell into an
  `ESGMetric` when the row label and cell both parse unambiguously.
- **Company / reporting year are user-supplied at upload**, not
  auto-detected from document content. 
- **Charts refuse to combine incompatible units** rather than doing a
  unit conversion — cross-report unit conventions vary enough (e.g. tCO2e
  vs. tCO2, KL vs. m³) that silent conversion is a bigger accuracy risk than
  a clear refusal message.