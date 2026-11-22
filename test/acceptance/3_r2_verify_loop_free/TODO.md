# R2 (loop-free token partial order) — intentionally not implemented

## What R2 requires

Spec section 3 R2: the token partial order, including `iter_args` carry
edges, must be acyclic.

## Why it is deferred

Cycles are **not expressible in the textual IR**, so R2 would be a check
that always passes today:

- Plain SSA / token use-def cycles cannot parse: MLIR requires an SSA
  value to be defined before use, so "A depends on B's token while B
  depends on A's token" fails at parse time with an undeclared-value
  error, long before any R2 analysis runs.
- `scf.for` `iter_args` carry edges are also acyclic by construction:
  a block argument's value comes from the *previous* iteration's
  `scf.yield`, so every carry edge points backwards in time.

The only way to build a cyclic token graph is to bypass the parser and
construct IR programmatically with `OpBuilder` (which performs no
dominance checking). No code path in the current pipeline does that —
the simulator only consumes parsed, compiler-emitted IR.

## When to implement

Implement R2 when an IR-transformation pass enters the pipeline
(scheduling, buffer rotation, or any builder-based IR generation). At
that point builder-produced cycles become a real failure mode and R2
gains a genuine defensive purpose. The implementation shape is a DFS /
SCC over token use-def edges plus `iter_args` carry edges, mirroring the
`tokenAncestors` helper already in
`lib/perf/Analysis/ScheduleLegalityVerifier.cpp`.

Testing note: a `.invalid.mlir` acceptance fixture for R2 is not
constructible — a cyclic module cannot be written in valid MLIR text.
Detection must be covered at the unit level by building the cyclic IR
with `OpBuilder` directly.
