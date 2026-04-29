# Intelligent Routing System (IRS)

An adaptive edge-cloud AI routing system that intelligently routes coding queries between local (Qwen Coder) and cloud (Claude Code) models based on task complexity.

## Quick Start

```bash
# Install dependencies
python3.11 -m pip install -r requirements.txt

# Interactive mode
./run.sh generate --interactive

# Health check
./run.sh check

# View stats
./run.sh stats
```

## Documentation

- **[PHASE1_README.md](PHASE1_README.md)** - Complete Phase 1 documentation (start here!)
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Full 7-phase project roadmap

## Current Status

**Phase 1**: ✅ Complete
- Intelligent routing between local and cloud models
- Conversation continuity support
- Rich CLI interface with statistics
- Performance tracking and logging

**Performance**: 62.5% local (free) / 37.5% cloud (5x faster)

---

For detailed setup, usage, and examples, see **[PHASE1_README.md](PHASE1_README.md)**
