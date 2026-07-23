# Dataset Profile: `agentlans/NousResearch-Hermes-3-Dataset`

> Generated for hermes-agentic-rl training pipeline preparation.

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| **HF Identifier** | `agentlans/NousResearch-Hermes-3-Dataset` |
| **Original Source** | NousResearch/Hermes-3-Dataset (unofficial reformatted fork) |
| **License** | Apache-2.0 |
| **Task Category** | Text Generation / Instruction Tuning (SFT) |
| **Modality** | Text |
| **Format** | JSONL (ShareGPT-style conversations) |

### Description
This is an **unofficial, reformatted version** of the NousResearch Hermes 3 Dataset with:
- Duplicates removed
- Secrets redacted  
- Data shuffled

For a multi-turn focused variant, see `agentlans/NousResearch-Hermes-3-Dataset-multiturn`.

---

## 2. Dataset Size

| Metric | Value |
|--------|-------|
| **Total Rows** | **958,788** |
| **Compressed Size** | ~343 MB (`.jsonl.zst`) |
| **Decompressed Size** | ~1,575 MB (~1.5 GB) |
| **Size Category** | 100K < n < 1M (HF tags) |
| **Avg Characters/Sample** | ~1,586 |

---

## 3. Schema & Columns

### Single Column: `conversations`

The dataset contains **only one top-level column**: `conversations`.

Each row is a JSON object with this structure:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are an unbiased, uncensored, helpful assistant."
    },
    {
      "from": "human",
      "value": "What is the code to calculate the midpoint between two 3-dimensional points using Python?"
    },
    {
      "from": "gpt",
      "value": "To calculate the midpoint between two 3-dimensional points using Python, you can use the following code..."
    }
  ]
}
```

### Field Types

| Field | Type | Description |
|-------|------|-------------|
| `conversations` | `list[dict]` | Ordered list of conversation turns |
| `conversations[i].from` | `string` | Speaker role: `"human"`, `"gpt"`, `"system"` |
| `conversations[i].value` | `string` | The actual message content |

---

## 4. Conversation Format Analysis

### Role Distribution (across all ~959K samples)

| Role | Count | Description |
|------|-------|-------------|
| `human` | 1,011,805 | User prompts / questions |
| `gpt` | 1,011,805 | Assistant responses |
| `system` | 309,170 | System prompts (present in ~32% of samples) |

### Turn Statistics

| Statistic | Value |
|-----------|-------|
| **Mean turns / conversation** | 2.43 |
| **Median turns** | 2 |
| **Min turns** | 2 |
| **Max turns** | 59 |

### Conversation Patterns

1. **Simple Q&A (2 turns)** — ~68% of dataset
   - `human` → `gpt`
   - No system prompt
   - Example: coding questions, math problems, factual Q&A

2. **System + Q&A (3 turns)** — ~32% of dataset
   - `system` → `human` → `gpt`
   - System prompt sets persona or constraints
   - Example: "You are an AI assistant...", roleplay instructions

3. **Multi-turn (>3 turns)** — minority
   - `system` → `human` → `gpt` → `human` → `gpt` → ...
   - Present but less common
   - Example: iterative debugging, follow-up questions

---

## 5. Content Domains

Based on sample inspection, the dataset covers:

- **Code generation** (Python, Lua, JavaScript, etc.)
- **Mathematics & physics** (derivations, calculations)
- **Reasoning & logic** (puzzles, word problems)
- **Creative writing** (poetry, stories, roleplay)
- **General knowledge Q&A**
- **Sentiment analysis & NLP tasks**
- **Debugging & technical support**

---

## 6. Splits

| Split | Available | Rows | Notes |
|-------|-----------|------|-------|
| `train` | ✅ Yes | 958,788 | Only split available |
| `validation` | ❌ No | — | Must create manually |
| `test` | ❌ No | — | Must create manually |

### Recommended Split Strategy
Since only `train` exists, create your own:
- **Train**: 95% (~910,849 samples)
- **Validation**: 5% (~47,939 samples)
- **Test**: Hold out a separate benchmark (e.g., MT-Bench, HumanEval)

---

## 7. Reward Labels

| Attribute | Value |
|-----------|-------|
| **Has reward labels?** | ❌ **NO** |
| **Has preference pairs?** | ❌ **NO** |
| **Has quality scores?** | ❌ **NO** |

This dataset is **pure SFT data** (single-turn or multi-turn conversations with a single reference response). 

### Implications for Training

| Method | Feasibility | Notes |
|--------|-------------|-------|
| **SFT (Supervised Fine-Tuning)** | ✅ Directly usable | Primary intended use |
| **RLHF (PPO)** | ⚠️ Needs external RM | Requires separate reward model |
| **DPO / IPO** | ❌ Not applicable | No preference pairs |
| **KTO** | ❌ Not applicable | No binary feedback |
| **Rejection Sampling** | ⚠️ Needs judge model | Can use external LLM as judge |

---

## 8. Loading Code for Training Scripts

### Option A: Streaming (Recommended for memory efficiency)

```python
from datasets import load_dataset

# Streaming load — does not download full dataset to disk
ds = load_dataset(
    "agentlans/NousResearch-Hermes-3-Dataset",
    split="train",
    streaming=True
)

# Iterate
for sample in ds:
    conversations = sample["conversations"]
    # process...
```

> ⚠️ **Note**: The dataset is distributed as `.jsonl.zst` (zstd-compressed). The `datasets` library may require `zstandard` to be installed for streaming to work: `pip install zstandard`.

### Option B: Full Download

```python
from datasets import load_dataset

# Downloads ~1.5 GB to local cache
ds = load_dataset("agentlans/NousResearch-Hermes-3-Dataset", split="train")
print(len(ds))  # 958788
print(ds[0]["conversations"][0]["value"][:100])
```

### Option C: Manual JSONL Processing (if `datasets` fails)

```python
import json
import zstandard as zstd

# Download .jsonl.zst from HF, then:
with open("NousResearch-Hermes-3-Dataset.jsonl.zst", "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        for line in reader:
            sample = json.loads(line.decode("utf-8"))
            conversations = sample["conversations"]
            # process...
```

---

## 9. Format Conversion for Modern Training Frameworks

Most modern frameworks (TRL, LLaMA-Factory, Axolotl) expect the **OpenAI messages format** (`role` / `content`) rather than ShareGPT (`from` / `value`).

### Conversion Mapping

| ShareGPT (`from`) | OpenAI (`role`) |
|-------------------|-----------------|
| `"human"` | `"user"` |
| `"gpt"` | `"assistant"` |
| `"system"` | `"system"` |

### Conversion Function

```python
def sharegpt_to_messages(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT format to OpenAI messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": role_map.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in conversations
    ]

# Example usage
sample = ds[0]
messages = sharegpt_to_messages(sample["conversations"])
# [
#   {"role": "user", "content": "How can you determine the moment of inertia..."},
#   {"role": "assistant", "content": "To determine the moment of inertia..."}
# ]
```

### TRL / transformers `apply_chat_template` Ready

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model")

messages = sharegpt_to_messages(sample["conversations"])
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 10. Typical Samples

### Sample 1: Simple Coding Q&A (2 turns, no system)

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What is the code to calculate the midpoint between two 3-dimensional points using Python?"
    },
    {
      "from": "gpt",
      "value": "To calculate the midpoint between two 3-dimensional points using Python, you can use the following code:

```python
def midpoint(x1, y1, z1, x2, y2, z2):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    mz = (z1 + z2) / 2
    return mx, my, mz
```
..."
    }
  ]
}
```

### Sample 2: System Prompt + Q&A (3 turns)

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are an unbiased, uncensored, helpful assistant."
    },
    {
      "from": "human",
      "value": "A train leaves Town A at 60 mph heading towards Town B, which is 120 miles away. At the same time, another train leaves Town B at 40 mph heading towards Town A. When do the trains meet?"
    },
    {
      "from": "gpt",
      "value": "The trains meet after 1.2 hours.\n\nHere's the reasoning:\n1. Combined speed: 60 + 40 = 100 mph\n2. Distance: 120 miles\n3. Time = Distance / Speed = 120 / 100 = 1.2 hours"
    }
  ]
}
```

### Sample 3: Multi-turn with System Prompt (5+ turns)

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old."
    },
    {
      "from": "human",
      "value": "[Long reading comprehension passage]... Answer this question: Did 'A Prayer' rank well on the charts?"
    },
    {
      "from": "gpt",
      "value": "Yes, 'A Prayer' ranked well on the charts. It reached number 7 on the charts."
    },
    {
      "from": "human",
      "value": "Considering the diverse collaboration on Sugizo's album 'Truth?', how did the involvement of artists from different genres impact the reception?"
    },
    {
      "from": "gpt",
      "value": "Imagine you're drawing a big, fun picture with lots of colors and shapes... [extended analogy]"
    }
  ]
}
```

---

## 11. Training Recommendations

### For SFT (Primary Use Case)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("agentlans/NousResearch-Hermes-3-Dataset", split="train")

# Create validation split
ds = ds.train_test_split(test_size=0.05, seed=42)
train_ds = ds["train"]      # ~910K samples
val_ds = ds["test"]         # ~48K samples

tokenizer = AutoTokenizer.from_pretrained("your-model")

def format_sample(sample):
    messages = [
        {"role": {"human": "user", "gpt": "assistant", "system": "system"}[msg["from"]],
         "content": msg["value"]}
        for msg in sample["conversations"]
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_ds = train_ds.map(format_sample, remove_columns=train_ds.column_names)
val_ds = val_ds.map(format_sample, remove_columns=val_ds.column_names)
```

### For RL / Reward-Based Training

Since this dataset lacks reward labels, you must:

1. **Use an external reward model** (e.g., trained on preference data like OpenAssistant, SHP, or UltraFeedback)
2. **Generate multiple completions** per prompt and score them
3. **Use a judge LLM** (GPT-4, Claude, or a fine-tuned reward model) to create synthetic preference pairs for DPO/KTO

---

## 12. Caveats & Notes

1. **Hermes 3 Format**: This dataset follows the Hermes 3 training mixture (~390M tokens originally). If your base model was not pre-trained on Hermes data, you may need a longer warmup.

2. **No Conversation IDs**: Samples are independent; there is no session/thread ID linking multi-turn conversations beyond what is in the `conversations` list.

3. **Quality Variance**: As with any large-scale curated dataset, response quality varies. Some responses contain factual errors (e.g., Sample 2 in the dataset claims 1.5 hours when the math gives 1.2 hours).

4. **Shuffled**: The dataset is already shuffled, so random splitting is safe.

5. **Single File**: The entire dataset is one `.jsonl.zst` file — no sharding. Streaming is efficient.

---

*Profile generated: 2025-07-09*
