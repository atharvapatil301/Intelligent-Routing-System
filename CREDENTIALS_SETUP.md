# Credentials Setup Guide

This guide helps you set up the required credentials for the Intelligent Routing System.

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your credentials** (see sections below)

3. **Verify setup:**
   ```bash
   ./run.sh check
   ```

---

## Required Credentials

### 1. PostgreSQL/Supabase Database (REQUIRED)

The system uses a PostgreSQL database with pgvector extension for vector similarity search.

**Default Setup (Supabase):**
```bash
POSTGRES_HOST=db.ezrhboaaipclzspfffxk.supabase.co
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_supabase_password_here  # ← ADD YOUR PASSWORD
```

**How to get credentials:**
- If using the default Supabase instance, ask the project owner for the password
- OR set up your own Supabase project:
  1. Go to https://supabase.com
  2. Create a free account
  3. Create a new project
  4. Get the database password from project settings
  5. Update all POSTGRES_* variables in `.env`

### 2. Groq API Key (REQUIRED for dataset generation)

Groq is used for LLM-as-judge evaluation during dataset generation (Phase 3).

```bash
GROQ_API_KEY=your_groq_api_key_here  # ← ADD YOUR API KEY
```

**How to get an API key:**
1. Go to https://console.groq.com
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into `.env`

**Note:** Only needed if you plan to generate training datasets. Not required for basic routing functionality.

---

## Optional Credentials

### 3. Anthropic API Key (OPTIONAL)

Only needed if you want to use the Anthropic API directly (not recommended - use Claude Code CLI instead).

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**How to get an API key:**
1. Go to https://console.anthropic.com
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key

**Note:** The system uses Claude Code CLI by default, which doesn't require an API key.

---

## Configuration Variables

### Ollama Configuration

```bash
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama URL
OLLAMA_MODEL=qwen2.5-coder:14b         # Default model
```

**Setup Ollama:**
1. Install Ollama from https://ollama.ai
2. Pull the model: `ollama pull qwen2.5-coder:14b`
3. Ensure Ollama is running: `ollama serve`

### Routing Configuration

```bash
PROMPT_LENGTH_THRESHOLD=100  # Token threshold for routing decisions
```

---

## Verification

After setting up your `.env` file, verify everything works:

```bash
# Check system health
./run.sh check

# Expected output:
# ✓ Ollama: Running
# ✓ Claude API: Configured
# Configuration shows all settings
```

---

## Security Notes

- **Never commit `.env` to version control** - it contains sensitive credentials
- The `.env` file is already in `.gitignore` to prevent accidental commits
- Use `.env.example` as a template (it has placeholder values)
- Keep your API keys and passwords secure
- Rotate credentials if they're exposed

---

## Troubleshooting

### Database Connection Issues

```bash
# Test connection manually
python3.11 -c "from src.vector_db import QueryVectorDB; QueryVectorDB()"
```

If you see errors:
- Verify `POSTGRES_PASSWORD` is correct
- Check network connectivity to Supabase
- Ensure firewall allows connections to port 5432

### Groq API Issues

```bash
# Test Groq connection
python3.11 -c "from groq import Groq; client = Groq(); print('✓ Groq configured')"
```

If you see errors:
- Verify `GROQ_API_KEY` is set correctly
- Check API key is active at https://console.groq.com
- Ensure you have API quota remaining

### Ollama Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull model if missing
ollama pull qwen2.5-coder:14b
```

---

## Environment Variables Summary

| Variable | Required | Purpose |
|----------|----------|---------|
| `POSTGRES_HOST` | Yes | Database host |
| `POSTGRES_PORT` | Yes | Database port |
| `POSTGRES_DB` | Yes | Database name |
| `POSTGRES_USER` | Yes | Database user |
| `POSTGRES_PASSWORD` | **Yes** | Database password |
| `GROQ_API_KEY` | **Yes*** | LLM-as-judge evaluation |
| `OLLAMA_BASE_URL` | Yes | Local model URL |
| `OLLAMA_MODEL` | Yes | Local model name |
| `ANTHROPIC_API_KEY` | No | Optional Anthropic API |
| `PROMPT_LENGTH_THRESHOLD` | Yes | Routing threshold |

\* Required only for dataset generation (Phase 3). Not needed for basic routing.
