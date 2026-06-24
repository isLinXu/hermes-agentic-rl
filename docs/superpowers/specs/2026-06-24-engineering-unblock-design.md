# 工程阻塞解锁设计

- 日期：2026-06-24
- 主题：真实闭环优先的工程阻塞修复
- 范围：`hermes-preflight`、`atropos-preflight`、`subprojects/hermes-agent` 中与 preflight 直接相关的确定性损坏
- 非范围：`eval-rl` 扩展、`Echo` 训练信号优化、无关重构

## 背景

当前仓库的最小离线训练闭环已经可运行，但真实 Hermes 闭环仍被工程问题阻塞。实际检查结果显示：

1. `subprojects/hermes-agent/hermes_cli/config.py` 存在确定性的语法损坏，导致 `hermes-preflight` 在导入 Hermes 子模块时直接崩溃。
2. `subprojects/tinker-atropos` 子模块未初始化，导致 `atropos-preflight` 无法完成导入验证。
3. Hermes 子项目存在若干依赖缺失时的导入失败，例如 `httpx`、`websockets` 等，但这些问题应在语法损坏修复后再逐层暴露并收敛。

目前最重要的不是扩大改动范围，而是把错误从“代码直接崩溃”收敛成“明确缺项、明确原因、可继续推进”。

## 目标

本批次目标只有三个：

1. 修复 `subprojects/hermes-agent` 中已知的确定性损坏，使 `hermes-preflight` 不再因语法错误直接失败。
2. 让 `hermes-preflight` 与 `atropos-preflight` 的输出尽可能收敛到结构化缺项，而不是一串难以定位的栈追踪。
3. 在不引入大规模行为变更的前提下，为后续真实闭环继续排障提供一个稳定起点。

## 成功标准

本批次完成后，至少满足以下标准：

1. `python -m hermes_agentic_rl.cli.main hermes-preflight` 不再因为 Hermes 子模块语法错误而崩溃。
2. 如果 `hermes-preflight` 仍失败，失败原因必须可以归类为：
   - 本地目录缺失
   - Python 模块缺失
   - 可选依赖缺失
   - 其他明确的导入失败
3. `python -m hermes_agentic_rl.cli.main atropos-preflight` 至少能清晰区分：
   - `subprojects/tinker-atropos` 未初始化
   - Python 依赖未安装
   - 真正的代码级异常

## 方案对比

### 方案 A：只修确定性损坏

直接修复 Hermes 子模块中已经确认损坏的代码，只处理会阻塞 preflight 的最短路径。

优点：

- 改动面最小
- 风险最小
- 最适合当前“先解阻塞”的目标

缺点：

- 无法一次性解决所有依赖问题
- 需要通过多轮 preflight 暴露下一层问题

### 方案 B：修损坏代码并同步补依赖

在修语法错误的同时，顺手安装或补齐当前缺失的运行依赖。

优点：

- 更接近“真实闭环直接可跑”

缺点：

- 变更面扩大
- 环境问题与代码问题会混在一起，不利于最小定位

### 采用方案

本批次采用 **方案 A**，但允许在验证阶段对少量明确缺失的基础依赖进行补齐，以便继续推进 preflight 验证。

## 详细设计

### 一、Hermes 子模块修复策略

只修复下列类型的问题：

1. 明确的语法错误
2. 因错误拼接、损坏插入、明显无效 token 导致的导入失败
3. 与 preflight 直接相关、且能明确证明为损坏而非业务逻辑选择的问题

不做下列事情：

1. 不重构 Hermes 子模块大文件
2. 不主动修改 Hermes 的正常行为逻辑
3. 不做与 preflight 无关的风格整理
4. 不将 Hermes 子项目整体升级或替换版本

### 二、Preflight 验证顺序

执行顺序固定如下：

1. 修复 `subprojects/hermes-agent/hermes_cli/config.py` 中已知语法损坏。
2. 运行 `hermes-preflight`。
3. 记录新的失败点。
4. 若新的失败仍属于“确定性代码损坏”，继续最小修复并重跑。
5. 若新的失败属于“依赖缺失”，将其归档为环境缺项，并仅在收益明显时补齐。
6. 最后运行 `atropos-preflight`，确认当前阻塞来自缺失子模块、缺失依赖还是代码错误。

### 三、Atropos/Tinker-Atropos 处理策略

当前 `git submodule status` 显示：

- `subprojects/atropos` 未初始化
- `subprojects/tinker-atropos` 未初始化

因此本批次对 Atropos 线的处理原则是：

1. 先验证 `atropos-preflight` 是否能清晰报告“子模块未初始化”。
2. 若报错信息不够清晰，可在本仓库 preflight 层补充更明确的缺项说明。
3. 不在本批次中大规模接入 Atropos 训练功能。

## 变更文件范围

预计允许修改的文件：

1. `subprojects/hermes-agent/hermes_cli/config.py`
2. `hermes_agentic_rl/integrations/hermes_preflight.py`
3. `hermes_agentic_rl/integrations/atropos_preflight.py`
4. 如有必要，少量与 preflight 直接相关的辅助文件

预计不修改的文件：

1. `hermes_agentic_rl/eval/*`
2. `hermes_agentic_rl/trainers/*`
3. `hermes_agentic_rl/envs/*`
4. `configs/*` 中训练配置

## 测试与验证

本批次不做大而全测试，只做与目标直接相关的验证：

1. `python -m hermes_agentic_rl.cli.main hermes-preflight`
2. `python -m hermes_agentic_rl.cli.main atropos-preflight`
3. 必要时补充一次 `git submodule status`
4. 如修改本仓库 preflight 代码，再做一次针对性静态读取或最小命令验证

## 风险与回退

### 风险

1. Hermes 子模块可能不止一处损坏，修复一个语法错误后会继续暴露更深层问题。
2. 部分导入错误可能来自环境未安装而不是代码损坏。
3. 直接修改子模块意味着后续同步上游时可能出现冲突。

### 回退策略

1. 每次只做最小改动，并在改动后立即重跑 preflight。
2. 不做大面积改动，降低回退成本。
3. 若某个问题无法在“最小修复”边界内安全处理，则停止扩大范围，并把它作为剩余缺项输出。

## 预期输出

本批次完成后，输出应包括：

1. 已修复的确定性损坏列表
2. 当前 `hermes-preflight` 结果
3. 当前 `atropos-preflight` 结果
4. 剩余缺项及建议下一步

## 自检结论

已完成自检，结论如下：

1. 文档中没有保留 `TODO`、`TBD` 等占位项。
2. 范围明确限定在工程阻塞解锁，不与 `eval-rl` 或 `Echo` 优化混杂。
3. 设计与实施顺序一致，且成功标准可直接通过命令验证。
4. 没有要求不可逆的大规模重构，符合当前最小修复目标。
