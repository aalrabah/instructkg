<div align="center">

<img src="banner.png" alt="InstructKG" width="100%">

### Automated Knowledge Graph Construction from Educational Content

*Building concept dependency graphs from lecture materials using multi-LLM architectures*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)


</div>

---

## Overview

**InstructKG** is an automated framework that extracts structured knowledge graphs from educational materials. It analyzes lecture slides, textbooks, and course materials to identify concepts and their pedagogical relationships — helping students understand learning paths and prerequisite dependencies.

### What It Does

Given educational PDFs, InstructKG:
1. **Extracts concepts** mentioned across lectures using LLMs
2. **Identifies roles** (definition, application, prerequisite) for each concept
3. **Clusters contexts** to find where concepts appear together
4. **Judges relationships** between concept pairs to build dependency graphs

### Why It Matters

- 📚 **For Students**: Understand which concepts to learn first and how topics connect
- 👨‍🏫 **For Educators**: Automatically generate course roadmaps and learning paths
- 🔬 **For Researchers**: Novel multi-LLM architecture for educational content understanding

---

## Architecture

InstructKG uses a **five-stage pipeline**:

```
PDFs → Chunking → Concept Extraction → Clustering → Pair Generation → Relation Judgment → Knowledge Graph
```

1. **Ingestion**: Converts PDFs to semantically meaningful chunks
2. **LLM Extraction**: Identifies concepts and their roles (definition/application/prerequisite)
3. **Clustering**: Groups similar contexts using UMAP + HDBSCAN
4. **Pair Packets**: Aggregates evidence for concept pairs from co-occurrences
5. **Relation Judgment**: LLM determines prerequisite/related/unrelated relationships

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/InstructKG.git
cd InstructKG

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Option 1: Run the full pipeline**

```bash
python main.py --data-dir ./lectures --out-dir ./output
```

**Option 2: Run specific steps**

```bash
# Only run concept extraction and relation judgment
python main.py --data-dir ./lectures --steps llm relations
```

**Option 3: Customize models and parameters**

```bash
python main.py \
  --data-dir ./lectures \
  --out-dir ./output \
  --llm-model meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 16 \
  --embedding-model all-MiniLM-L6-v2 \
  --min-cluster-size 3
```

---

## Pipeline Stages

### 1. Ingestion (`ingest`)
Converts PDF lectures into chunks with metadata.

**Output**: `chunks.jsonl`

### 2. Concept Extraction (`llm`)
Uses LLMs to extract concepts and classify their roles.

**Models supported**: GPT-4, Llama 3, Qwen 2.5, Mistral

**Output**: `mentions.jsonl`, `concept_cards.jsonl`

### 3. Clustering (`clustering`)
Groups similar contexts where concepts appear.

**Techniques**: Sentence embeddings → UMAP → HDBSCAN

**Output**: `context_clusters.jsonl`

### 4. Pair Packet Generation (`pairpackets`)
Aggregates evidence for concept pairs from chunk and cluster co-occurrences.

**Output**: `pairpackets.jsonl`

### 5. Relation Judgment (`relations`)
LLM determines if concept pairs have prerequisite relationships.

**Output**: `relations.jsonl` (final knowledge graph edges)

---

## Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/` | Folder containing PDF lectures |
| `--out-dir` | `out/` | Output directory for results |
| `--llm-model` | `gpt-4o-mini` | LLM model for extraction |
| `--batch-size` | `8` | Batch size for LLM calls |
| `--embedding-model` | `all-MiniLM-L6-v2` | Model for embeddings |
| `--min-cluster-size` | `2` | Minimum cluster size for HDBSCAN |
| `--max-pairs` | `None` | Limit number of concept pairs (for testing) |
| `--steps` | `all` | Pipeline steps: `ingest llm clustering pairpackets relations` |

---

## Example

```bash
# Process CS 225 (Data Structures) lectures
python main.py --data-dir ./cs225_lectures --out-dir ./cs225_kg

# Output will include:
# - 1,247 concepts extracted
# - 3,891 concept pairs evaluated
# - 423 prerequisite relationships identified
```

**Sample extracted relationships:**
- `arrays` → `linked lists` (prerequisite)
- `trees` → `binary search trees` (prerequisite)
- `graphs` → `graph traversal algorithms` (prerequisite)

---

## Project Structure

```
InstructKG/
├── main.py                 # Main pipeline orchestrator
├── ingest.py              # PDF → chunks conversion
├── llm.py                 # Concept extraction + role classification
├── clustering.py          # Context clustering (UMAP + HDBSCAN)
├── pairpackets.py         # Evidence aggregation for pairs
├── relation_judger.py     # Relation classification
├── config.py              # Configuration defaults
├── requirements.txt       # Python dependencies
├── data/                  # Input PDFs (your lectures)
└── out/                   # Output files
    ├── chunks.jsonl
    ├── mentions.jsonl
    ├── concept_cards.jsonl
    ├── context_clusters.jsonl
    ├── pairpackets.jsonl
    └── relations.jsonl    # Final knowledge graph
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
