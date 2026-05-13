# Hermes Agentic RL - Comprehensive Optimization Plan

## Executive Summary

This project trains agentic RL (GRPO/PPO) models using a dual-path architecture:
1. **train-rl** path: Local GRPO/PPO training with session-sidecar + replay + self-evolution data export
2. **online-cycle** path: Deployed Hermes with session-sidecar + replay + self-evolution data export

### Critical Code Debt

| Issue | Severity | Impact | Files |
|-------|----------|--------|-------|
| E501 line length disabled | Medium | Readability | Multiple |
| B008 function call default argument | High | Shared-state bugs | `on_policy.py`, `curriculum.py` |
| B023 loop variable closures | High | Subtle bugs | `on_policy.py` |

## Optimization Strategy

### Phase 1: Fix Critical Debt (B008)

**Problem**: `field(default_factory=dict)` creates a new dict each time, causing shared-state bugs.

**Solution**: Replace with `field(default_factory=SHARED_DICT)` where `SHARED_DICT` is a module-level dict created ONCE.

**Impact**:
- Fixes B008 violations in `on_policy.py` and `curriculum.py`
- Eliminates shared-state bugs from `field(default_factory=dict)`

### Phase 1.5: Fix B023 (Loop Variable Closures)

**Problem**: Lambda closures in loops capture the loop variable at the last iteration, not the current one.

**Solution**: Refactor to avoid lambda closures in loops. Use explicit function definitions or `functools.partial` instead.

**Impact**:
- Fixes B023 violations in `on_policy.py`
- Makes loop variable bindings explicit and predictable

## Implementation Plan
