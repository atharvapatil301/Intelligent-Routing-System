# Intelligent Routing System (IRS)

An adaptive edge-cloud AI routing system that intelligently routes coding queries between local (Qwen Coder) and cloud (Claude Code) models based on task complexity.

## Quick Start

```bash
# Install dependencies
python3.11 -m pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your credentials (POSTGRES_PASSWORD, GROQ_API_KEY)

# Interactive mode
./run.sh generate --interactive

# Health check
./run.sh check

# View stats and performance
./run.sh stats
./run.sh dashboard

# Evaluate routing system
./run.sh evaluate data/datasets/dataset_test.jsonl

# Compare routing strategies
./run.sh compare data/datasets/dataset_test.jsonl
```

## Documentation

- **[PROGRESS.md](PROGRESS.md)** - Complete progress report with all phases, features, and running instructions
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Full 7-phase project roadmap

## Current Status

**Phase 1**: ✅ Complete
- Intelligent routing between local and cloud models
- Conversation continuity support
- Rich CLI interface with statistics
- Performance tracking and logging

**Phase 2**: ✅ Complete
- Advanced feature extraction (19 features + embeddings)
- Semantic embeddings (384-dim using sentence-transformers)
- Complexity signal detection (concurrency, algorithms, reasoning)
- PostgreSQL + pgvector for similarity search (hosted on Supabase)
- Enhanced routing strategy

**Phase 3**: ✅ Complete
- Dataset generation infrastructure
- Dual-model evaluation (Qwen + Claude)
- 100+ seed prompts across 10 categories
- LLM-as-judge evaluation using Groq (llama-3.1-8b-instant)
- Train/val/test split generation
- JSONL dataset format
- Automated dataset generation and training pipeline

**Phase 4**: ✅ Complete
- Neural network routing model (145K parameters)
- 3-layer feedforward architecture with dropout and batch normalization
- 86.67% test accuracy on routing decisions
- ML-based routing (now default)
- Enhanced rule-based routing (fallback)
- Configurable decision threshold
- Model persistence and loading

**Phase 5**: ✅ Complete
- Multi-objective cost optimization (accuracy, latency, cost)
- Confidence-based routing with automatic escalation
- Threshold tuning and optimization
- Strategy comparison framework

**Phase 6**: ✅ Complete
- Comprehensive evaluation framework
- Dataset evaluation with test files
- Live model comparison
- Performance dashboard CLI command
- Automated report generation
- 66.7% cost reduction vs always-cloud baseline

---

For detailed setup, usage, examples, and complete progress report, see **[PROGRESS.md](PROGRESS.md)**
