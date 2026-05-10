# Intelligent Routing System (IRS)

An adaptive edge-cloud AI routing system that intelligently routes coding queries between local (Qwen Coder) and cloud (Claude Code) models based on task complexity.

## Quick Start

```bash
# Install dependencies
python3.11 -m pip install -r requirements.txt

# Set your Supabase database password
export POSTGRES_PASSWORD="your_password_here"

# Interactive mode
./run.sh generate --interactive

# Health check
./run.sh check

# View stats
./run.sh stats
```

## Documentation

- **[PROGRESS.md](PROGRESS.md)** - Complete progress report with all phases, features, and running instructions (✅ Phase 1-3 Complete)
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
- Heuristic evaluation method
- Train/val/test split generation
- JSONL dataset format

**Phase 4**: ✅ Complete
- Neural network routing model (145K parameters)
- 3-layer feedforward architecture with dropout and batch normalization
- 86.67% test accuracy on routing decisions
- ML-based routing strategy
- Configurable decision threshold
- Model persistence and loading

**Next**: Phase 5 - Optimization & Cost Analysis

---

For detailed setup, usage, examples, and complete progress report, see **[PROGRESS.md](PROGRESS.md)**
