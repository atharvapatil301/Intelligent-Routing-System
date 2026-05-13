# Intelligent Routing System - Implementation Plan

## Project Overview

**Goal**: Build an adaptive edge-cloud AI routing system that intelligently decides whether to route coding queries to a local model (Qwen Coder) or cloud model (Claude Code) based on task complexity, cost, and accuracy optimization.

**Value Proposition**: Reduce cloud API costs by 40-60% while maintaining comparable code quality through intelligent, learned routing decisions.

---

## System Architecture

```
User Query (Coding Prompt)
   ↓
Feature Extraction Layer
   ├── Structural Features (length, complexity keywords)
   ├── Semantic Features (embeddings)
   └── Historical Features (similarity to past queries)
   ↓
Routing Decision Engine (ML Model)
   ├── Predicts: P(local_success)
   └── Decides: local vs cloud
   ↓
┌─────────────────────┬─────────────────────┐
│  Local: Qwen Coder  │  Cloud: Claude Code │
│  (fast, cheap)      │  (accurate, costly) │
└─────────────────────┴─────────────────────┘
   ↓
Response + Performance Logging
   ↓
Feedback Loop (continuous learning)
```

---

## Implementation Phases

### Phase 1: Baseline System (Week 1)
**Objective**: Get basic routing working with simple heuristics

**Components**:
- [ ] Set up Qwen Coder locally
- [ ] Integrate Claude Code API
- [ ] Implement rule-based routing
  - If prompt_length < 100 tokens → local
  - If keywords like "optimize", "design", "architecture" → cloud
  - Else → local
- [ ] Basic CLI interface for testing
- [ ] Logging infrastructure (prompt, model_used, response, latency)

**Deliverables**:
- Working CLI tool that routes to either model
- Baseline performance metrics to beat later

---

### Phase 2: Feature Engineering (Week 2)
**Objective**: Extract meaningful features from queries

**Features to Implement**:

#### Basic Features:
- Prompt length (tokens, characters)
- Number of functions/classes requested
- Presence of code blocks
- Question type detection (implementation vs debugging vs design)

#### Advanced Features (Differentiation):
- **Embeddings**: Use small embedding model (sentence-transformers)
  - Convert query to semantic vector
  - Captures problem type beyond keywords

- **Complexity Signals**:
  - Reasoning keywords: "why", "derive", "prove", "optimize"
  - Concurrency mentions: "thread-safe", "concurrent", "race condition"
  - Algorithm complexity: "O(n log n)", "dynamic programming"

- **Similarity to Past Queries**:
  - Build vector database of past queries
  - For new query, find k-nearest neighbors
  - Calculate success rate of local model on similar queries

**Deliverables**:
- Feature extraction module
- Vector database for query embeddings
- Feature vector format standardized

---

### Phase 3: Dataset Creation (Week 2-3)
**Objective**: Create training data for routing model

**Data Collection Strategy**:

#### Seed Dataset Sources:
- HumanEval (coding problems with test cases)
- LeetCode-style problems
- Custom prompts across difficulty levels
- Target: 500-1000 samples

#### Data Generation Process:
1. For each prompt, run BOTH models
2. Collect outputs from Qwen and Claude
3. Evaluate which was better using:
   - **Unit tests** (primary method for coding tasks)
   - **LLM-as-judge** (for non-testable queries)
   - **Heuristic scoring** (syntax errors, completeness)

#### Dataset Schema:
```json
{
  "prompt": "string",
  "embedding": [float],
  "features": {
    "length": int,
    "num_functions": int,
    "has_concurrency": bool,
    "complexity_keywords": [string],
    "similarity_to_failures": float
  },
  "qwen_output": "string",
  "claude_output": "string",
  "qwen_success": bool,
  "claude_success": bool,
  "label": 0 or 1  // 0 = local sufficient, 1 = cloud needed
}
```

#### Data Augmentation:
- Paraphrase prompts (same problem, different wording)
- Difficulty scaling (add/remove constraints)
- Synthetic variations using GPT

**Deliverables**:
- 500+ labeled samples
- Train/val/test split (70/15/15)
- Failure memory database

---

### Phase 4: ML Routing Model (Week 3)
**Objective**: Replace rule-based routing with learned model

**Model Options** (start simple, iterate):

#### Option A: Logistic Regression / XGBoost
- Fast to train
- Interpretable
- Good baseline

#### Option B: Neural Network (if time permits)
- Small feedforward network
- Input: feature vector
- Output: P(local_sufficient)

**Training Process**:
- Input: [embeddings + structural features + similarity scores]
- Output: Binary classification (local=0, cloud=1)
- Loss: Binary cross-entropy
- Optimization: Balance precision/recall based on cost function

**Decision Rule**:
```python
if P(local_sufficient) > threshold:
    route_to_local()
else:
    route_to_cloud()
```

**Threshold Tuning**:
- Adjust based on cost vs accuracy tradeoff
- Higher threshold → more cloud calls (safer, costly)
- Lower threshold → more local calls (risky, cheap)

**Deliverables**:
- Trained routing model
- Model evaluation metrics
- Threshold optimization analysis

---

### Phase 5: Optimization & Cost Function (Week 4)
**Objective**: Optimize for multi-objective goals

**Cost Function**:
```
score = accuracy - λ1 * latency - λ2 * cost
```

**Tunable Parameters**:
- λ1: latency penalty weight
- λ2: cost penalty weight
- Threshold for routing decision

**Advanced Features**:
- **Confidence-based routing**: Run local first, escalate if uncertain
- **Semantic caching**: Store responses for similar queries
- **Online learning**: Update model from production feedback

**Deliverables**:
- Optimized cost function
- A/B testing framework
- Performance dashboard

---

### Phase 6: Evaluation (Week 4)
**Objective**: Prove the system works

**Comparison Matrix**:

| Strategy        | Cloud Cost | Latency | Accuracy | Notes |
|-----------------|-----------|---------|----------|-------|
| Always Local    | $0        | 50ms    | 65%      | Fast but unreliable |
| Always Cloud    | $100      | 800ms   | 95%      | Accurate but expensive |
| **Smart Router**| **$35**   | **200ms**| **90%** | **Best balance** |

**Metrics to Track**:
- % cloud calls avoided
- Cost reduction (dollar amount)
- Accuracy degradation (if any)
- Latency distribution (p50, p95, p99)
- False negative rate (local failed when shouldn't have)
- False positive rate (cloud used unnecessarily)

**Deliverables**:
- Comprehensive evaluation report
- Charts and visualizations
- Statistical significance testing

---

### Phase 7: Product Interface (Week 5, Optional)
**Objective**: Make it usable beyond CLI

**Interface Options**:

#### Option 1: CLI Tool (MVP)
```bash
irs "implement binary search tree"
# Output: code + model used + latency
```

#### Option 2: VS Code Extension (Best UX)
- Hotkey trigger (Ctrl+K)
- Inline code generation
- Status bar showing routing decision
- Cost/latency dashboard

#### Option 3: API Service
```python
POST /generate
{
  "prompt": "...",
  "prefer_local": true  # optional hint
}

Response:
{
  "code": "...",
  "model_used": "qwen",
  "latency_ms": 120,
  "confidence": 0.87
}
```

**Deliverables**:
- Production-ready interface
- Usage analytics
- User feedback mechanism

---
