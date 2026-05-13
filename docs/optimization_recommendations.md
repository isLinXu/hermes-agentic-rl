# Optimization Recommendations for hermes-agentic-rl

## Summary of Issues Identified

### Critical Debt

1. **B008: Mutable Default Argument (High Priority)**
   - Files: `hermes_agentic_rl/trainers/on_policy.py`, `hermes_agentic_rl/envs/curriculum.py`
   - Issue: `field(default_factory=dict)` creates a new dict per instance, causing shared-state bugs
   - Fix: Replace with `field(default_factory=SHARED_DICT)` where `SHARED_DICT` is a module-level dict created once

2. **B023: Loop Variable Closures (Medium Priority)**
   - Files: `hermes_agentic_rl/trainers/on_policy.py`
   - Issue: Lambda closures in loops capture the loop variable at the last iteration, not the current one
   - Fix: Avoid lambda closures in loops; use explicit function definitions or `functools.partial`

### Recommended Changes

#### Phase 1: Fix B008 (Quick Wins)

**File: `hermes_agentic_rl/trainers/on_policy.py`**

```python
# Current (B008 violation):
iters: list[dict[str, Any]] = field(default_factory=dict)

# Fixed:
iters: list[dict[str, Any]] = field(default_factory=lambda: list(SHARED_DICT.keys()))
```

Apply the same fix to all `field(default_factory=dict)` occurrences in the file.

**File: `hermes_agentic_rl/envs/curriculum.py`**

```python
# Current (B008 violation):
recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=32))

# Fixed:
recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=32), repr=False)
```

#### Phase 2: Fix B023 (Medium Priority)

**File: `hermes_agentic_rl/trainers/on_policy.py`**

Search for all lambda definitions inside loops and refactor them into standalone functions or `functools.partial` calls.

**File: `hermes_agentic_rl/envs/curriculum.py`**

Same as above - replace lambda with def or functools.partial.

#### Phase 3: Code Quality (Low Priority but Recommended)

- Enable E501 line length checking (currently disabled)
- Remove unnecessary B008/B023 suppressions
- Replace `isinstance(x, X | Y)` with Python 3.10+ compatible form: `isinstance(x, (X, Y))`

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| B008 violations | 1 | 0 | ✅ Fixed |
| B023 violations | ~5 | 0 | ✅ Fixed |
| E501 violations | ~3 | 0 | ✅ Fixed (if enabled) |
| Code clarity | Lambdas in loops | Explicit functions | ✅ Improved |

### Implementation Steps

1. **Fix B008 in `on_policy.py`** (1 change)
2. **Fix B008 in `curriculum.py`** (1 change)
3. **Fix B023 in `on_policy.py`** (~5 changes)
4. **Fix B023 in `curriculum.py`** (~5 changes)
5. **Enable E501 checking** (optional, can be done later)
6. **Remove B008/B023 suppressions**
7. **Replace `isinstance(x, X | Y)`** (~3-5 changes)
8. **Run linter to verify**
9. **Run tests**

### Risk Assessment

- **Low risk**: B008/B023 fixes are localized, no architectural changes
- **Medium risk**: E501 enabling may expose previously-hidden issues
- **High risk**: `isinstance(x, X | Y)` replacement requires Python 3.10+ (already satisfied)
