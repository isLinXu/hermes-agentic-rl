# Comprehensive Optimization Plan for hermes-agentic-rl

## Executive Summary

This project implements agentic RL (GRPO/PPO) training via a dual-path architecture:
1. **train-rl**: Local GRPO/PPO training with session sidecar + replay + self-evolution data export
2. **online-cycle**: Deployed Hermes with session sidecar + replay + self-evolution data export

## Critical Code Debt

| Issue | Severity | Impact | Affected Files |
|-------|----------|--------|----------------|
| E501 line length checking disabled | Medium | Readability | Multiple modules |
| B008 function call as default argument | High | Shared-state bugs | `on_policy.py`, `curriculum.py` |
| B023 loop variable closures | High | Subtle bugs | `on_policy.py` |

## Optimization Strategy

### Phase 1: Fix Critical Debt (B008)

**Problem**: `field(default_factory=dict)` creates a new dict per instance, causing shared-state bugs.

**Solution**: Replace with `field(default_factory=SHARED_DICT)`, where `SHARED_DICT` is a module-level dict created ONCE.

**Impact**:
- Eliminates B008 violations in `on_policy.py` and `curriculum.py`
- Fixes shared-state bugs from `field(default_factory=dict)`

### Phase 1.5: Fix B023 (Loop Variable Closures)

**Problem**: Lambda closures in loops capture the loop variable at the last iteration, not the current one.

**Solution**: Refactor to avoid lambda closures in loops. Use explicit function definitions or `functools.partial` instead.

**Impact**:
- Fixes B023 violations in `on_policy.py`
- Makes loop variable bindings explicit and predictable

## Phase 2: Code Quality (Optional but Recommended)

- Enable E501 line length checking (currently disabled)
- Replace `isinstance(x, X | Y)` with Python 3.10+ compatible form: `isinstance(x, (X, Y))`

## Implementation Order

1. Fix B008 in `on_policy.py` (1 change)
2. Fix B008 in `curriculum.py` (1 change)
3. Fix B023 in `on_policy.py` (~5 changes)
4. Fix B023 in `curriculum.py` (~5 changes)
5. Enable E501 checking (optional, can be deferred)
6. Replace `isinstance(x, X | Y)` with Python 3.10+ compatible form (~3-5 changes)

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| B008 violations | 1 | 0 |
| B023 violations | ~5 | 0 |
| E501 violations (if enabled) | ~3 | 0 |
| Python compatibility | 3.10+ required | 3.10+ compatible |

## Risk Assessment

- **Low risk**: B008/B023 fixes are localized, no architectural changes
- **Medium risk**: E501 enabling may surface previously-hidden issues
- **High risk**: `isinstance(x, X | Y)` replacement requires Python 3.10+ runtime

## Next Steps

1. Implement B008 fixes
2. Implement B023 fixes
3. Run linter (ruff) to verify changes
4. Run tests
5. Enable E501 checking (optional)
6. Replace isinstance with Python 3.10+ compatible form
