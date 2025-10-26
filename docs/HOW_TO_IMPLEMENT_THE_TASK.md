# How to Implement a Task with AI Agents

A structured, multi-agent approach to implementing complex tasks with quality assurance built in at every step.

## Core Principles

### 1. Agent Isolation

**Use a new agent (new conversation) for each specialized phase to prevent context leak.**

Each agent has a focused role and shouldn't be burdened with information from other phases. This keeps agents sharp, reduces token costs, and prevents confusion from irrelevant context.

### 2. Human-in-the-Loop for Critical Decisions

The main idea is to **specify the task, plan, decomposition strategy, and acceptance criteria** (unit tests, metric thresholds) upfront. These can be drafted with AI assistance, but **it is crucial to review and validate them personally** before proceeding to implementation.

**Only after validation should you delegate implementation to AI agents.** This ensures the right problem is being solved the right way.

______________________________________________________________________

## The Five-Phase Workflow

### Phase 1: Task Refinement

**Agent Role:** Task Refine Agent
**Location:** `docs/tasks/{task_path}/task.md`

**What happens:**

1. Create a new folder structure: `docs/tasks/{task_path}/`
2. Write an initial `task.md` with your high-level task description
3. The Task Refine Agent will:
   - Ask clarifying questions
   - Refine the task description
   - Add concrete details and constraints
   - Document assumptions and requirements

**Output:** A clear, detailed task description that serves as the single source of truth.

______________________________________________________________________

### Phase 2: Task Decomposition

**Agent Role:** Planning Agent
**Location:** `docs/tasks/{task_path}/` (tree structure)

**What happens:**
Decompose the task into smaller, manageable subtasks using one of two strategies:

- **Incremental decomposition:** Break into small, sequential tasks
- **Iterative decomposition:** Break into prototypes with small diffs between iterations

**Critical principle:** Each step must be **automatically testable** to allow agents to improve code without manual human verification.

**Tree Structure:**
The decomposition forms a tree, mirrored in the folder structure:

```
docs/tasks/{task_path}/
├── task.md                          # Root task
├── subtask_1/
│   ├── task.md                      # Subtask 1 description
│   ├── sub_subtask_1a/
│   │   └── task.md                  # Leaf node
│   └── sub_subtask_1b/
│       └── task.md                  # Leaf node
└── subtask_2/
    └── task.md                      # Subtask 2 description
```

- **Internal nodes:** Have a `task.md` describing the current subtask + subfolders for children
- **Leaf nodes:** Have only a `task.md` (no subfolders)

**Output:** A complete task tree with each node clearly defined and testable.

______________________________________________________________________

### Phase 3: Architecture Design

**Agent Role:** Architecture Agent
**Location:** `docs/tasks/{task_path}/{node_path}/ARCHITECTURE.md`

**What happens:**
For each tree node (bottom-up approach):

1. **Leaf nodes first:** Design architecture for the smallest units

   - Define classes, functions, their signatures
   - Specify docstrings, arguments, return types
   - Indicate where they should be stored (file paths)
   - **No implementation yet** — only contracts

2. **Parent nodes:** Merge child architectures and add integration details

   - Take architecture from children
   - Add glue code, interfaces, and coordination logic
   - It's OK to modify child architectures if needed for coherence

3. **Root node:** Final, complete architecture of the entire task

**Output:** `ARCHITECTURE.md` for each node, describing what to build without how to build it.

______________________________________________________________________

### Phase 4: Acceptance Criteria & Test Implementation

**Agent Role:** Criteria Agent
**Location:** Tests created in appropriate test directories

**What happens:**
For each tree node, define and **implement** automatic validation:

- **Unit tests:** Test individual components
- **Integration tests:** Test interactions between components
- **Metric thresholds:** Define success criteria (accuracy > 0.9, latency \< 100ms, etc.)
- **Behavioral tests:** Verify expected behavior on edge cases

Since the architecture is already fixed, the agent knows exactly what interfaces to test.

**Output:**

- Test files implemented and ready to run
- Clear pass/fail criteria for each node
- Tests initially fail (red) — this is expected

______________________________________________________________________

### Phase 5: Implementation

**Agent Role:** Coding Agent
**Location:** Implementation files as specified in architecture

**What happens:**
For each tree node, implement in a **test-driven cycle**:

```
Code → Test → Code → Test → Code → ...
```

The agent stops when **all tests pass** (green).

**Critical constraints:**

- **Provide clear scope:** The agent should only edit files related to its assigned node
- **Lock tests:** The agent **must not** modify test files or unrelated code
- **Isolated work:** Each node is implemented independently

**Implementation order:**

- Start with leaf nodes
- Move up the tree to parent nodes
- Complete with root node integration

**Output:** Fully implemented, tested, and passing code for the entire task tree.

______________________________________________________________________

## Summary

| Phase        | Agent              | Input                | Output                       | Review Required |
| ------------ | ------------------ | -------------------- | ---------------------------- | --------------- |
| 1. Refine    | Task Refine Agent  | Initial task idea    | Detailed `task.md`           | ✅ Yes           |
| 2. Decompose | Planning Agent     | Task description     | Task tree structure          | ✅ Yes           |
| 3. Design    | Architecture Agent | Subtasks             | `ARCHITECTURE.md` per node   | ✅ Yes           |
| 4. Test      | Criteria Agent     | Architecture         | Implemented tests (failing)  | ✅ Yes           |
| 5. Implement | Coding Agent       | Architecture + Tests | Working code (tests passing) | Optional\*      |

\* *Manual review optional in Phase 5 because automated tests verify correctness*

______________________________________________________________________

## Benefits of This Approach

1. **Quality assurance built in:** Tests defined before code
2. **Clear boundaries:** Each agent has a focused role
3. **Incremental validation:** Each node is verified independently
4. **Reduced context leak:** Fresh agent for each phase
5. **Human oversight:** Critical decisions reviewed before expensive implementation
6. **Automatic verification:** Agents iterate until tests pass
7. **Tree structure:** Natural parallelization opportunities for leaf nodes

______________________________________________________________________

## Example Workflow

Let's say you want to implement a new data augmentation pipeline:

**Task documentation structure:**

```
docs/tasks/data_augmentation_pipeline/
├── task.md                                    # Phase 1: Refined task
├── ARCHITECTURE.md                            # Phase 3: Overall architecture
├── transform_base/
│   ├── task.md                                # Phase 2: Subtask definition
│   └── ARCHITECTURE.md                        # Phase 3: Component architecture
├── image_transforms/
│   ├── task.md
│   └── ARCHITECTURE.md
└── integration/
    ├── task.md
    └── ARCHITECTURE.md
```

**Test structure (Phase 4):**

```
tests/test_tasks/test_data_augmentation_pipeline/
├── test_transform_base/
│   ├── test_base_transform.py
│   └── test_transform_registry.py
├── test_image_transforms/
│   ├── test_resize.py
│   ├── test_normalize.py
│   └── test_compose.py
└── test_integration/
    ├── test_pipeline_integration.py
    └── test_end_to_end.py
```

Each folder gets its own specialized agent attention, building from leaves to root.

______________________________________________________________________

## Tips for Success

1. **Start small**: Begin with a simple task to validate the workflow before tackling complex features
2. **Be specific in Phase 1**: The better the initial task description, the better all downstream work
3. **Review architectures carefully**: Changes in Phase 3 are cheap; changes in Phase 5 are expensive
4. **Write strict tests**: Loose tests in Phase 4 lead to poor implementation in Phase 5
5. **Scope aggressively**: Tell the Coding Agent exactly what files it can and cannot touch
6. **Use version control**: Commit after each phase to enable easy rollback
7. **Iterate on process**: Adjust the workflow based on what works for your team
8. **Document decisions**: Keep notes on why certain architectural choices were made

______________________________________________________________________

## References

- `docs/SPEC_TEMPLATE.md` — For writing specifications
- `docs/ARCHITECTURE.md` — For understanding system architecture patterns
- `docs/TESTING.md` — For testing best practices
- `AGENTS.md` — For single-agent execution patterns

