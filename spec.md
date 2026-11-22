# perf Dialect Core Semantics Specification

This document specifies the `perf` dialect: IR semantics, verification rules, the hardware configuration model, the cost model contract, and the simulator's execution semantics.

Conventions used throughout:

- "Must" states a normative requirement; violating it is an implementation defect. "May" marks something permitted but not required.
- IR fragments use the dialect's real syntax. Names inside attribute parameters (space names, resource kinds) are strings and must be quoted.
- Units: time in cycles, capacity in SRAM rows, bandwidth in bytes per cycle.

---

## 1. Concepts and Governing Principle

### 1.1 Governing Principle

IR correctness depends only on the partial order. Performance depends on the schedule. A schedule can be replaced without regenerating the IR.

This principle organizes the whole document: chapter 2 defines the IR that expresses the partial order, chapter 4 defines the rules that verify it, and chapter 5 defines how performance is derived on top of it.

### 1.2 Partial Order

The partial order `≺` is formed by two kinds of SSA edges: use-def edges on ordinary data values, and use-def edges on token values.

$$
\prec = \text{transitive closure}(\text{SSA use-def} \cup \text{token use-def})
$$

Only ops that are incomparable under the partial order may run in parallel. If the partial order degenerates into a total order, there is no parallelism left to exploit.

### 1.3 Tokens

`!perf.token` carries the happens-before relation and expresses only genuine data and memory dependencies:

- dependencies on immutable tensors and scalars are expressed by SSA use-def and need no token;
- RAW, WAR and WAW dependencies on mutable memrefs and banks must be completed by tokens;
- a token does not express a scheduling preference such as "I would like this to run first";
- a token does not express resource contention order. Resource order belongs to the schedule.

### 1.4 Schedule

A schedule is a total order over the ops on a single resource, supplied by the `perf.schedule` annotation. It is not part of token semantics and may be replaced without affecting IR correctness.

When a module carries no `perf.schedule`, the default schedule on an `in_order` resource is the program text order of the block. This default is part of the specification: the verifier constructs the schedule by this rule on unannotated input, and a compiler that needs any non-program order must state it explicitly with `perf.schedule`. The default rule decides only the relative order within one resource; it creates no cross-resource happens-before.

### 1.5 Resources

A resource is an object with the capacity to serve requests over time. There are three classes:

| Class | Instances | Referenceable by IR placement | Role |
|---|---|---|---|
| Executor | DMA channels, arrays | Yes | Executes ops on a timeline |
| Link | DRAM bus, shared write port | No | Appears only in cost model demands |
| Bank port | Read/write port of each bank | No | Ledger created by the engine from the memspace declaration |

Placement is an attribute of an op stating which resource instance it is bound to, written `#perf.res<kind, index>`.

Mutual exclusion between resources is expressed by how the cost model's phase chain and demands land on resource ledgers (see 5.3), never by a slot count.

### 1.6 Dual Semantics of Storage

Every memory space carries both spatial and temporal semantics:

- **Spatial (capacity and address).** A buffer occupies the local row-address interval `[base_row, base_row + rows)`. Bank membership is derived from the row address through the mapping declared in the configuration, by default `bank(row) = row / rows_per_bank`, so one logical buffer may span several banks. Capacity is a legality question, checked per bank at peak by rule R5, and never enters the simulator's cycle accounting.
- **Temporal (port contention).** Each access expands over its address interval into read or write row requests against one or more banks. Concurrent requests mapped to the same bank and the same direction are arbitrated by the priority declared in the configuration; losers are deferred, and the resulting stall is attributed by the engine to the point of contention (see 5.5).

### 1.7 Role of the Hardware Configuration

The hardware configuration describes the structural parameters and calibration corrections of the target NPU. It takes no part in IR semantic correctness, but it determines the numbers in a performance prediction:

- the verifier uses it to check legality (R1 connectivity, R5 capacity);
- the cost model uses it to compute each op's EstimatedTiming;
- the simulator uses it to derive resource timelines.

Running the same IR against a different configuration leaves semantics unchanged and the performance prediction different. The verifier and the simulator must share one configuration; they must not operate on different hardware assumptions.

---

## 2. IR Semantics

### 2.1 Common Contract

Ops with side effects obey a common contract:

- consume 0..N `!perf.token` values, with token dependencies written inside brackets;
- produce exactly one completion token;
- carry the placement attribute `resource = #perf.res<kind, index>`;
- distinguish read and write memref operands explicitly.

Completion uses a single-token model: one token means every access this op makes to every relevant memory has completed. This is a conservative model.

### 2.2 `perf.alloc`

Produces a root buffer, which is the root of alias analysis (see chapter 3):

```mlir
%buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0, 32>>
```

`#perf.space<name, base_row, rows>` states that the buffer occupies the local row-address interval `[base_row, base_row + rows)` inside space `name`. Shape and element type are carried by the memref type and are not duplicated as attributes. Bank is not part of the type; bank membership is derived from the row address through the configuration's bank mapping.

The verifier checks two things: that `rows` agrees with memref shape × element width ÷ the configuration's `row_width_bytes` (rounded up), and that the interval does not exceed the capacity of the space.

Only static row-address intervals are accepted, with `base_row` and `rows` constant, which makes bank membership statically decidable.

### 2.3 `perf.dma_load` and `perf.dma_store`

Explicit transfers, with direction distinguished by the op name:

```mlir
%la = perf.dma_load %A, %spad_a [%free] {
  bytes = 512,
  resource = #perf.res<"dma_read", 0>
} : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

%st = perf.dma_store %acc_c, %C [%c0] {
  bytes = 1024,
  resource = #perf.res<"dma_write", 0>
} : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token
```

Requirements:

- the `dst` of `dma_load` must be an on-chip memref carrying `#perf.space`, and the `src` of `dma_store` must be on-chip;
- `bytes` feeds the cost model and validation, and must agree with memref shape × element width;
- layout information needed to model burst, bank and strided access is taken from the memref type;
- a transfer is two-sided: `dma_load` raises an external bus read demand together with an on-chip bank write demand, and `dma_store` raises an on-chip bank read demand together with an external bus write demand. The two sides are parallel bottlenecks and must never be summed serially (see 5.7).

### 2.4 `perf.array_init`

Establishes the initial state of the PE accumulator registers. The memref operand is optional, and its presence decides the semantics:

```mlir
// No operand: synthesize zeros through the zero-fill path, reading no SRAM.
%p0 = perf.array_init {
  m = 16, n = 16,
  resource = #perf.res<"array", 0>
} : () -> !perf.token

// With an operand: read the existing partial sums of that region, creating a RAW dependency.
%p1 = perf.array_init %acc_c [%t_wb] {
  m = 16, n = 16,
  resource = #perf.res<"array", 0>
} : (memref<256xi32, #perf.space<"acc", 0, 16>>, !perf.token) -> !perf.token
```

- The operandless form reads no SRAM, declares no Read effect and creates no RAW edge; when it may issue is constrained only by the array resource and the schedule.
- The form with an operand pours existing partial sums or a bias into the PE registers and raises an acc bank read demand (see 5.6).
- This op initializes only the PE accumulator registers and touches no other array state.

### 2.5 `perf.compute`

Array computation:

```mlir
%c0 = perf.compute %spad_a, %spad_b, %acc_c [%la, %lb, %p0] {
  m = 16, n = 16, k = 32,
  writeback = true,
  resource = #perf.res<"array", 0>
} : (memref<512xi8, #perf.space<"spad", 0, 32>>,
     memref<512xi8, #perf.space<"spad", 32, 32>>,
     memref<256xi32, #perf.space<"acc", 0, 16>>,
     !perf.token, !perf.token, !perf.token) -> !perf.token
```

Attributes:

- `m`, `n`, `k`: the true execution granularity of this instruction. All three must be positive.
- `writeback`: whether this instruction includes the final result writeback stage (PE registers → SRAM).
- `dataflow`: optional. When omitted, the configuration's fixed dataflow applies; when given explicitly, the value must be one of the configuration's `supported_dataflows`.

Execution semantics: compute accumulates onto the array's current state (`PE += A × B`), and there is no second mode. The PE state is established by `perf.array_init` or by the preceding compute, and its origin must be traceable along the token chain (rule R6).

This yields the rule governing C's SRAM traffic: **C is read only by `array_init` and written only by writeback.** A compute never raises a C bank read. This is a structural fact, not a special case driven by an attribute.

A and B input accesses bind to the array input window, and the C writeback binds to the writeback window; both are expressed at SRAM row request granularity (see 5.6).

Instruction granularity is the compiler's responsibility: the simulator must not split `k = 1024` into 64 instances of `k = 16`. If the target ISA has an instruction granularity of 16, the compiler must emit that many computes.

### 2.6 `perf.fence`

An event join that occupies no execution resource:

```mlir
perf.fence [%st0, %st1]
```

### 2.7 `perf.token.init`

Produces an initial token and takes no operands. This op and loop-carried tokens must not be removed by DCE.

### 2.8 `perf.prim`

A general extension point for ops that have no dedicated contract yet. To take part in precise cost modeling it must implement the corresponding `CostModelInterface`; otherwise only the default coarse model applies.

---

## 3. Alias Model

### 3.1 Root Buffers and Views

- A **root buffer** is produced by `perf.alloc` and carries a concrete memory space, a row-address interval, and an alloc site.
- A **view** is produced by `memref.subview`, `reinterpret_cast` and similar ops. It must be statically traceable to a unique root and must yield a static byte interval, which the configuration's row width converts into a row interval.

Hard constraints: roots are never merged; a view must be traceable to a unique root; when the root or the interval cannot be determined statically, the verifier rejects the program.

### 3.2 Conflict Test

Given two accesses `u` and `v`:

```text
conflict(u, v) ⟺ root_u == root_v ∧ [a_u, b_u) ∩ [a_v, b_v) ≠ ∅
```

Distinct roots never conflict; within one root the test reduces to interval intersection.

### 3.3 Bank Contention Is a Separate Dimension

Two accesses contend for bank `b` exactly when both of their row-address intervals cover `b` after the configured mapping and the directions agree, or when the bank is single-ported and the directions are arbitrary.

Buffers with distinct roots may well map to the same bank. That is not a conflict, since they carry no data dependency; it is contention in the performance dimension, resolved by engine arbitration (see 5.5).

### 3.4 Alignment with Generic MLIR Passes

Ordering truth is: SSA use-def for immutable values, tokens for mutable memory. MemoryEffects exist only to keep generic passes from mishandling side-effecting ops; they are not the ordering truth.

An op carrying tokens must therefore:

- declare correct read and write MemoryEffects;
- not be reordered by generic canonicalize, CSE, DCE or LICM according to text order or effect order;
- be reordered only by the dialect's own scheduling passes, and only while respecting the partial order.

---

## 4. Verification Rules

In the rules below, the partial order `≺` is the transitive closure of SSA use-def and token use-def (see 1.2).

### R1 Reachability

For every op, the memory space holding each of its memref operands must lie within the connectivity of that op's placement resource. Bank is derived from the address and takes no part in the connectivity test.

### R2 Acyclicity

The token partial order must be acyclic, including `iter_args` carry edges.

### R3 Schedule Legality

The schedule on every resource must be a linear extension of the partial order `≺`. Without a `perf.schedule` annotation the schedule is taken from the default rule in 1.4; with an annotation it is taken from the annotation.

A violation manifests as replay deadlock: by the readiness condition in 5.5 the op at the queue head waits for a predecessor to complete, while that predecessor can never issue because it is not at a queue head. R3 turns this class of runtime hang into a compile-time rejection.

The rule constrains only conflicts in which a write participates. Pure read conflicts carry no data dependency, so their order is purely a performance dimension and is left as search space for the schedule. Port and bank contention is arbitrated by priority in the engine and belongs to the same dimension.

### R4 Write Conflict

If two accesses conflict (see 3.2), at least one is a write, and the two are incomparable under the partial order, the program is rejected.

### R5 Capacity (Per-Bank Peak)

Whether two buffers may be live at the same time is decided by the partial order, independent of text order. R5 must hold conservatively over every legal linear extension of the partial order: the looser the partial order, the larger the set that may be simultaneously live, and the more conservative the check.

The check proceeds as:

1. derive from the partial order an upper bound on the set of buffers that may be live at any instant (maximum antichain, or interval-overlap formulation);
2. project each live buffer onto per-bank row occupancy through its row-address interval and the configuration's bank mapping;
3. verify, for **every bank** of every memory instance, that the rows the live set maps onto that bank sum to no more than `rows_per_bank`.

Checking only the total capacity of a space is insufficient. When the total fits but the peak on one bank does not, a banked scratchpad cannot realize that buffer plan.

### R6 Array State Provenance

The PE state of every `perf.compute` must have an origin: walking back along the token chain must reach a `perf.array_init` or another `perf.compute`. Accumulating onto array register state of unknown provenance is not allowed.

In addition, `array_init`, which overwrites PE register state, must be ordered against the previous tile's writeback: that writeback must complete before the init begins. This WAR dependency is carried by the token partial order, and its absence is a rejection.

### Rules and the Schedule

R1, R2, R4 and R5 depend only on the partial order. Replacing `perf.schedule` does not require re-verifying those four; it requires only verifying that the new schedule is still a legal linear extension (R3).

---

## 5. Simulation Semantics and Cost Model

### 5.1 Layering

| Layer | Content | Role |
|---|---|---|
| Layer 0 | Partial-order DAG | Expresses genuine dependencies only |
| Layer 1 | `perf.schedule` | Gives each resource a total order over ops, a linear extension of the partial order |
| Layer 2 | Hardware configuration and cost model | Computes EstimatedTiming for each dynamic instance |

The simulator takes four inputs: the partial-order DAG, which decides where execution must wait across resources; the schedule, which decides the replay order within a resource; the hardware configuration, which describes structure including bank mapping and arbitration priority; and the cost model, which maps a dynamic instance to an EstimatedTiming.

The DAG cannot be dropped. With a schedule but no DAG, the simulator would assume every resource can run back to back and would systematically erase cross-resource bubbles.

What the simulator orders is not static nodes but dynamic instances:

```text
OpInstance  = (static_op, iteration_vector)
OpSegment   = a run of OpInstances that can be aggregated
LoopSummary = a loop region treatable in closed form or by recurrence
```

For example, a static node `%cp` inside a loop body corresponds at runtime to `(%cp, i=0)`, `(%cp, i=1)`, and so on.

### 5.2 EstimatedTiming

The cost model returns neither a single cycle count nor a phase chain of fixed durations. It returns a quote consisting of nominal windows plus resource demands:

```text
EstimatedTiming {
  phases: [Phase],              // ordered phase chain; the skeleton is invariant
  isolated_lower_bound          // completion bound if no demand ever queued
}

Phase {
  nominal_window,               // cycles of this phase in isolation, free of contention
  demands: [Demand]             // resource demands raised by this phase
}
```

Semantics:

- `nominal_window` is the phase length in isolation. After engine arbitration the actual window may stretch, and the stretch is exactly the contention stall, attributed to a specific point of contention.
- `isolated_lower_bound` serves reporting and sanity checks and is never a scheduling input.
- A phase may carry no demands, which models a pure time tail: it pushes completion later without occupying anything.

### 5.3 The Four Landing Modes of a Demand

A Demand is not one structure with switch fields but four types. They differ in **landing mode**, and the landing mode is an intrinsic property of the resource: every Demand on one resource is necessarily of the same type, and the configuration declares each resource's arbitration policy as either `fifo` or `priority`.

```text
Demand {
  resource,                     // resource name
  tag                           // reporting label, e.g. "A" / "B" / "BUS" / "W"
}

BlockQueueDemand  : Demand { span }         // queues FIFO, then owns the whole span
BlockOffsetDemand : Demand { span }         // raises the phase window floor, without owning the span
BlockFollowDemand : Demand { }              // occupancy follows the phase's actual window
CycleDemand       : Demand { span, prio }   // contends cycle by cycle under priority
```

| Type | Settled | Occupancy semantics | Typical resource |
|---|---|---|---|
| `BlockQueueDemand` | At declaration, O(1) | Queues on the resource's `free_at`; once served, owns the resource for the whole `span` | DMA channel, array issue |
| `BlockOffsetDemand` | At declaration, O(1) | Raises the phase window floor to the resource's free time and yields a completion bound `span` cycles later, without blocking other concurrent demands | DRAM bus |
| `BlockFollowDemand` | Backfilled once the actual window is known | Occupies for exactly the phase's actual window, which cannot be known at declaration | Array (occupancy stretches in step when bank contention stretches the phase) |
| `CycleDemand` | Placed on the timeline and judged cycle by cycle by the cursor | Contends under `prio` against other requests on the same bank and direction, accumulating granted cycles until `span` is reached | Bank read port, bank write port |

`span` is a cycle count, converted by the cost model from `count × granularity ÷ port width`. The conversion is pure arithmetic and belongs inside the cost model; the engine sees only `span` and knows nothing of request counts or granularity.

`prio` is needed only by `CycleDemand`. Lower numbers win, and the values come from the configured arbitration priority table (see 6.3).

For a DMA load with an outstanding-capable channel, a shared bus and an on-chip bank write:

```text
[ Phase(issue):    nominal = request_issue,
                   demands = [BlockQueueDemand(dma_read#0, "ISS", span = request_issue)],
  Phase(transfer): nominal = max(bus transfer, bank write),      // parallel bottlenecks
                   demands = [BlockOffsetDemand(dram_bus,      "BUS", span = bus cycles),
                              CycleDemand(spad bank write port, "W",  span = bank cycles, prio = dma)] ]
```

Derived reporting figures: an op's occupancy on resource `r` is the sum of the occupancy records its phases produce on `r` (see 5.5), and its total duration is `done − start`. A stall is not a static cost term; it is produced by event simulation.

### 5.4 CostModelInterface

```text
CostModelInterface:
  CostModel(hw_config)                          // injected at construction, immutable thereafter
  evaluate(op) -> EstimatedTiming
  evaluateAggregate(segment) -> AggregateEstimatedTiming
```

`AggregateEstimatedTiming` serves steady-state loops and runs of like instances.

The hardware configuration is injected at construction rather than passed per call: within one simulation the configuration is immutable, so the only variable of `evaluate` is the op instance. Derived constants such as initialization bandwidth and the request distribution implied by the bank mapping are therefore computed once at construction and cached.

The return value depends only on the op instance and the immutable configuration captured at construction. The cost model must hold no state that changes as the simulation progresses, and it is unaware of any queueing or contention state.

### 5.5 Event Semantics

#### Declaration

The engine declares at **phase** granularity, interleaved, rather than walking a whole phase chain at the moment an op becomes ready:

```text
op x becomes ready   → declare phase 0 of x
phase i of x ends    → declare phase i+1 of x
last phase of x ends → record done(x) and wake successors
```

Interleaved declaration is what makes cross-op contention computable: a phase's demands enter the arbitration candidate set only once declared, and they contend only when they meet other already-declared demands on the timeline.

Readiness condition:

```text
ready(x) ⟺ every partial-order predecessor of x is done
           ∧ x == head(schedule[eng(x)])
```

When several ops become ready at the same instant, they are declared in program order. The engine is a replayer and never picks an op out of a ready set on its own.

#### The Declaration Invariant

Let phase `p` be declared at cursor time `now`. Its window floor is:

```text
window_start(p) = max(now, end of previous phase, done of all predecessors)   // first phase uses predecessors only
```

**No declaration may have a `window_start` earlier than its declaration time `now`.** Cycle-by-cycle judgment happens only at the cursor, and judged history never changes: the engine has no rollback, no invalidation cascade, and no versioning.

The invariant matters most when an op becomes ready late because of its schedule position. Suppose `B`'s dependency completes at t=10 but `B` sits behind `A` in the queue, and `A` is declared only at t=100. Then `B`'s window floor must be 100 and must not fall back to 10; otherwise `B` would be settled in a past that has already been judged.

#### Landing Within a Phase

At declaration a phase lands in this order:

```text
1. for each BlockOffsetDemand(r):       window_start = max(window_start, free_at(r))
2. for each BlockQueueDemand(r, span):  served = max(window_start, free_at(r))
                                        free_at(r) = served + span
   actual_start = max(window_start, every BlockQueue served time)
3. for each BlockOffsetDemand(r, span): free_at(r) = max(free_at(r), actual_start + span)
4. for each CycleDemand:                place on the timeline with actual_start as its activation floor
5. completion floor = max(actual_start + nominal_window, completion of every Block-class demand)
```

The phase's actual window is `[actual_start, completion)`, where `completion` additionally waits until every `CycleDemand` of this phase has accumulated `span` granted cycles. Once the actual window is known, each `BlockFollowDemand` is backfilled: it occupies its resource over `[actual_start, completion)` and updates that resource's `free_at`.

A `BlockFollowDemand` must start at `actual_start` rather than `window_start`: during the queueing wait the resource is not held by this op, and counting that period would inflate occupancy and delay successors.

#### Cycle-by-Cycle Arbitration

On a `priority` resource, which is a bank port, one judgment is made per cursor cycle:

```text
candidates = CycleDemands on that port that are active (activation floor ≤ now) and unfinished
winner     = the candidate minimizing (prio, declaration sequence number)
the winner is granted this cycle and its granted count rises by one; it finishes on reaching span
every other candidate is deferred this cycle, accruing one cycle of contention stall
                       attributed to (blocked demand, winner)
```

The declaration sequence number is a globally monotonic counter, so among equal priorities the earlier declaration wins. Since ops that become ready at the same instant are declared in program order, arbitration among equal priorities is determined uniquely by program order.

Because judgment is per cycle, arbitration is preemptive: when a higher-priority request activates midway through a lower-priority request's service, it takes the port from that cycle onward, and the lower-priority request resumes accumulating after the higher-priority one finishes.

On a single-ported bank, reads and writes share one port and enter the same candidate set.

#### Fast-Forwarding

The outcome of cycle-by-cycle arbitration can change only at two kinds of instant: a winner finishes, or the candidate set changes because a new demand activates or a finished phase triggers a new declaration. The cursor may therefore advance in one step of:

```text
k = min( remaining cycles of the current winner on each priority resource,
         time of the next pending event − now )
```

Within `[now, now + k)` the winner on each port is invariant, so granting k cycles at once is identical to granting one cycle k times. Fast-forwarding is exact, not an approximation. Pending events, meaning phase completions and demand activations, are the cursor's only stopping points.

When no `priority` resource has a candidate, the cursor jumps straight to the next event time. Having neither candidates nor events while ops remain unfinished is a deadlock, and the engine must report it and list the unfinished ops.

#### Equivalence of FIFO Settlement

A `fifo` resource settles at declaration through `free_at` in O(1) and never reaches the timeline. This is strictly equivalent to settling cycle by cycle on the timeline: a fifo resource is non-preemptive, and by the declaration invariant the declaration order is the queueing order.

Consequently the queueing order on a fifo resource is fixed by declaration order alone and is unaffected by any bank priority.

#### Occupancy Records

Every landing produces an occupancy record `Reservation(resource, owner, start, done)`, where `owner` is `op name:tag`. These records are the only raw material for reporting and error attribution, so they must be faithful to the intervals actually served:

- a record's `start` is the cycle at which the demand **first actually receives service**, not its activation floor. When a request activates but keeps losing, the waiting period is not occupancy;
- the occupancy of a preempted `CycleDemand` is **fragmented**, and each contiguous granted segment produces its own record. A single interval must not span from first grant to completion, or the preempted cycles would be counted as occupancy.

Two checkable invariants follow: occupancy records on one port never overlap, and a port's total occupancy equals the sum of the `span` values of its demands and never exceeds the elapsed cycles.

#### Timestamps and Stalls

- `start(x)`: the instant the first demand of x's first phase actually receives service. When x waits in a fifo queue, `start(x)` is later than that phase's `window_start`.
- `done(x)`: the `completion` of x's last phase.
- Stalls fall into three classes:

| Class | Arises when | Attribution |
|---|---|---|
| Dependency stall (bubble) | An op waits for a partial-order predecessor | Idle interval on the resource |
| Queueing stall | A `BlockQueueDemand` has `served > window_start` | `queue:<resource name>` |
| Contention stall | A `CycleDemand` is not granted in some cycle | The winner's `op name:tag` |

The resulting behavior: when the next op on a resource may start follows from that resource's `free_at`; when an op depending on `x` may start follows from `done(x)`; and both the sequential queueing of one op across sub-resources (a DMA transfer phase waiting on the bus, a writeback phase waiting on the acc write port) and cycle-level contention on bank ports (A and B on one bank, DMA against the array) emerge naturally from replay.

### 5.6 Array Cost Contract

The array's EstimatedTiming expresses access at SRAM row request granularity, and its phase chain is organized by the following windows:

```text
[ Phase(input):     nominal = boundary_input_window(A/B) → last_mac,
                    demands = [BlockFollowDemand(array#0, "ARR"),
                               CycleDemand(spad bank read, "A", prio = execute_a),
                               CycleDemand(spad bank read, "B", prio = execute_b)],
  Phase(output):    nominal = output_completion_window,
  Phase(writeback): nominal = writeback_window,
                    demands = [CycleDemand(acc bank write, "C", prio = execute_wb)],
  Phase(tail):      nominal = post_input_tail (no demands, a pure time tail) ]
```

The array uses `BlockFollowDemand` rather than `BlockQueueDemand`: when bank contention stretches the input phase, the array's occupancy must stretch in step, and the amount of stretch is unknown at declaration.

Window definitions:

| Window | Meaning |
|---|---|
| `boundary_input_window(A/B)` | Window in which the first A/B row requests enter the array input path |
| `first_mac` / `last_mac` | Timestamps of the first and last MAC, delimiting the stream segment |
| `output_completion_window` | Window in which array output, including reduction, completes |
| `writeback_window` | Window writing results back to acc or scratchpad; C writeback requests bind here |
| `post_input_tail` | Trailing segment after the input path drains but before output completes |

Rules:

- `array_init` is a separate op and does not join the compute phase chain. Its duration follows from the resident data volume, the initialization bandwidth, and the array structure.
- Whether `array_init` raises a bank read is decided by operand presence: without an operand it takes the zero-fill path and raises none; with an operand it must read the bank, for existing partial sums or a bias.
- A compute's A and B read requests bind to the input phase, covering the boundary window and the stream segment, and its C writeback binds to `writeback_window`.
- A compute never raises a C bank read (see 2.5).
- The stream segment follows from the temporal loop count and the initiation interval. The cadence between `first_mac` and `last_mac` follows from array topology, skew, PE latency, inter-tile pipelining, and the reduction path. When `inter_tile_pipelined = false`, the output phase folds into array occupancy.
- Writeback follows jointly from the result byte count, the array egress, the reduction network, the write port, and accumulator write bandwidth. When the acc write port is contended among `array_init`, writeback and DMA, the queueing is produced by engine arbitration.
- When A, B and C accesses map to the same bank, the cost model quotes only their respective `CycleDemand`s; the engine arbitrates by the configured priority table.
- The concrete rules generating access streams belong to the target contract. The first such contract is Gemmini, documented in `gemmini-target-profile.md`. This specification does not define a general access-stream model for arbitrary memref layouts.

Window cycle counts are not supplied directly as base structural parameters. The backend derives them from structural parameters, and only calibration terms may correct them.

### 5.7 DMA Cost Contract

A DMA transfer is a two-sided parallel process, and its EstimatedTiming must quote demands on both sides:

```text
dma_load:  external bus read demand + on-chip bank write demand
dma_store: on-chip bank read demand + external bus write demand

[ Phase(issue):    nominal = request_issue,
                   demands = [BlockQueueDemand(channel, "ISS", span = request_issue)],
  Phase(transfer): nominal = max(bus-side transfer, bank-side transfer),
                   demands = [BlockOffsetDemand(dram_bus, "BUS", span = bus cycles),
                              CycleDemand(bank port, "W", span = bank cycles, prio = dma)] ]
```

- The two sides are parallel bottlenecks: the transfer phase's nominal window takes the larger of the two and must never sum them serially. Contention stretching either side pushes completion later.
- The issue phase uses `BlockQueueDemand` on the channel named by placement: the channel is held for the whole issue phase, so a following DMA must queue.
- The bus uses `BlockOffsetDemand`: it supplies a completion bound and raises the phase start, but does not hold the whole span exclusively. When the bank side stretches under contention, the bus has already been released and should not block other transfers.
- The bank port uses `CycleDemand`, derived from the row-address interval of the destination or source through the mapping.
- A channel without outstanding support degenerates to a single phase. Whether reads and writes share the bus is declared by the configuration.
- Inputs: byte count, shape and stride, burst granularity, channel width, fixed request latency, outstanding limit, and the target bank mapping.

### 5.8 Mixed-Granularity Simulation

An `scf.for` should not merely be interpreted iteration by iteration as a syntax tree. The simulator should first build a `LoopRegion`:

```text
LoopRegion {
  trip_count
  body_static_ops
  iter_args token carry
  per-resource schedule template
  cross-iteration recurrence
  per-iteration EstimatedTiming function
}
```

and then handle it in one of three modes:

1. **Per-instance event driven**, for prologue, epilogue, edge tiles and irregular regions.
2. **Steady-state fast-forward**, for the middle region where schedule and EstimatedTiming repeat.
3. **Segmented aggregation**, handling prologue, steady region, edge tiles and epilogue separately.

Steady-state fast-forwarding is safe when: the trip count is computable; the schedule template repeats; cross-iteration dependencies form a fixed recurrence; EstimatedTiming is independent of the iteration index or summable in closed form; buffer capacity and aliasing have been checked by the verifier; there is no irregular control flow; and edge tiles have been carved out separately.

When those conditions do not hold, fall back to lazy expansion rather than expanding every iteration at once.

An aggregated block is represented as:

```text
SteadyBlock {
  iteration_range = [i0, i1)
  per_resource_occupancy
  first_ready
  last_done
  boundary_tokens
}
```

Internal dynamic tokens need not all be materialized; it suffices that boundary tokens and externally visible dependencies remain correct.

### 5.9 Simulator State and Main Loop

Base state:

```text
cursor now (monotonically increasing)

per schedule queue:
  declaration pointer (advances in program order)

per fifo resource:
  free_at time
  list of occupancy records

per priority resource (bank port):
  set of unfinished CycleDemands
  list of occupancy records

event heap:
  (time, sequence number, kind) — kind is "phase complete" or "demand activates"

completion table: done(x)
actual start table: start(x)
stall ledger: cycles accumulated per (blocked party, attribution)
```

Main loop:

```text
loop:
    1. process every due event at time ≤ now (which may chain into new phase declarations)
    2. advance each queue's declaration pointer: declare the first phase of a head op
       once all its predecessors are done; ties at one instant go in program order
    3. all ops done → finish
    4. collect the current winner on each priority resource; with no winner at all,
       jump the cursor to the next event time (no events means deadlock)
    5. compute the fast-forward step k and advance the cursor
    6. grant k cycles to each winner; a demand whose granted count reaches span finishes,
       and a phase settles once its count of unfinished demands reaches zero
```

Steps 1 and 2 must not be swapped: a due event may make a new op ready, so events must be fully processed before the declaration pointers advance, or ops becoming ready at that instant would be deferred by a round.

Two degenerate cases: when a resource is strictly exclusive and its demands never contend, quoting and landing reduce to a phase chain of fixed durations; and when a `priority` resource holds a single candidate for a long stretch, the fast-forward step equals that demand's entire remaining span, which is equivalent to landing the whole segment at once.

---

## 6. Hardware Configuration

The configuration describes structure only and supplies no complete cycle formula.

### 6.1 The Two Configuration Layers

The hardware configuration is defined by two independent top-level ops:

- a **design point** (`perf.target_def` with `#perf.design`), holding the physical structural parameters that DSE sweeps;
- a **simulation prior** (`perf.prior_def` with `#perf.prior`), holding calibration results shared across one machine family.

```mlir
perf.target_def @systolic_npu_v1 #perf.design<
  version = 5,
  target_name = "systolic_npu_v1",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >,
    #perf.memspace<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64
    >
  ],
  dma_read_ports = 1,
  dma_write_ports = 1,
  arrays = 1,
  dma_channel = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  array_uarch = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 1, tile_cols = 1,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  share_dram_bus = true
>

perf.prior_def @gemmini_base #perf.prior<
  version = 1,
  target_name = "gemmini_base",
  calibration = #perf.calibration<
    extra_fill_cycles = 0,
    extra_drain_cycles = 0,
    extra_preload_cycles = 0,
    extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>
```

### 6.2 Deriving the Contract

The vocabulary of executors and links is not written by hand. `TargetModel::build` projects it from the design point and materializes it as `perf.contract_def`, observable through `perf --dump-contract`. Program IR refers only to `#perf.res<kind, index>`.

Bank ports work the same way: the engine opens a ledger for each bank's read and write port from the memspace declaration. They may appear in Demands and in reports, and they may not be referenced by IR placement.

### 6.3 Arbitration Priority

The role names in the arbitration priority table, namely `execute`, `dma_load`, `dma_store` and `zero_writer`, are client classes at the level of the target contract. When the cost model quotes a Demand it assigns the op kind to a role, and the engine judges by the table.

One bank mapping is currently defined, `high_bits`, meaning `bank = row / rows_per_bank`.

---

## 7. Acceptance Criteria

Acceptance has three levels:

1. **Per-module error.** Compare DMA, array, bank arbitration, bubbles and writeback against the reference implementation separately, rather than comparing only total cycles.
2. **Error across a stated coverage range.** Report mean and maximum error within a stated range of workloads, dataflows, array sizes and tile sizes.
3. **Ranking confidence.** Promise that a ranking is reliable only when the predicted difference between two schedules exceeds the calibration uncertainty.

The target is no more than 10% error on key scenarios, and it must always be stated together with its coverage range. The specification does not require that any two schedules be ranked consistently.

---

## 8. Supported Scope

### 8.1 IR

`perf.alloc`, `perf.token.init`, `perf.dma_load`, `perf.dma_store`, `perf.array_init`, `perf.compute`, `perf.fence`.

### 8.2 Hardware

One systolic array; a fixed OS dataflow; a banked scratchpad and accumulator, with row-address intervals, bank mapping and priority arbitration on ports; one DMA read channel; one DMA write channel; fixed DRAM latency and bandwidth.

### 8.3 Simulation

A single completion token; mutual exclusion expressed through phase chains and the four demand landing modes; interleaved declaration at phase granularity; the `start`, `free_at` and `done` timestamps; per-instance event-driven replay; cycle-by-cycle priority arbitration on bank ports with exact fast-forwarding.

---

## 9. Gemmini Correlation Study

### 9.1 Role of Each Reference Tool

| Tool | Use |
|---|---|
| Spike | Functional correctness only; never a performance reference |
| Verilator / VCS | Primary cycle-accurate calibration source |
| FireSim | Cycle-accurate comparison for larger end-to-end workloads |
| Genus / Innovus | Area, frequency and power data; outside this simulator's scope |

Gemmini's storage contract, arbitration rules and compute access-stream generation are documented in `gemmini-target-profile.md`, which is authoritative for configuration instances.

### 9.2 The Correlation Loop

1. Pick a kernel, starting from a 64×64×64 matmul.
2. Generate IR and a schedule with this compiler, run this simulator, and obtain `C_sim`.
3. Construct an equivalent computation on Gemmini, run Verilator, VCS or FireSim, and obtain `C_rtl`.
4. Compare DMA, array, bank arbitration, bubbles and writeback separately.
5. Correct EstimatedTiming and the configuration.
6. Report the error within a stated coverage range.

There are two ways to construct the equivalent computation. The end-to-end route has this compiler emit Gemmini RoCC code and run it under a cycle-accurate simulator; it is the strictest and the most work. The hand-written route writes Gemmini C code whose tile, DMA and compute order mirrors the schedule in this IR, which suits quick calibration.

### 9.3 Order of Error Attribution

1. DMA fixed latency and bandwidth.
2. SRAM row request streams and bank/port arbitration, including A/B/C priority on one bank and DMA against the array.
3. Byte counts and ports of `array_init` and writeback.
4. Array stream and initiation interval, and the `first_mac`/`last_mac` cadence.
5. Transposer and OS row-order corrections.
6. Cross-resource bubbles.

---

## 10. Rationale for a Native MLIR Implementation

| Rationale | Detail |
|---|---|
| The simulator needs IR structure | `scf.for`, `iter_args`, tokens and schedules are all native MLIR information |
| Verifier and simulator share analyses | Partial order, buffer provenance and resource placement need not be written twice |
| The schedule is an IR attribute | Autotuning a new schedule stays within one process |
| LoopRegion analysis is reusable | Steady-state fast-forwarding depends on structured loop information |
| The cost model attaches as an interface | `CostModelInterface` binds directly to ops and to the target backend |
| The contract is shared with code generation | One compilation result can lower to real hardware and also feed the simulator |

The implementation shape that follows: event-driven replay preserves semantics in irregular regions, structured analysis fast-forwards regular regions for speed, and compiler, verifier and simulator share one IR and one hardware model.

---

## Appendix A. Complete Example

### A.1 Hardware Configuration (target.mlir)

```mlir
perf.target_def @systolic_npu_v1 #perf.design<
  version = 5,
  target_name = "systolic_npu_v1",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >,
    #perf.memspace<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64
    >
  ],
  dma_read_ports = 1,
  dma_write_ports = 1,
  arrays = 1,
  dma_channel = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  array_uarch = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 1, tile_cols = 1,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  share_dram_bus = true
>
```

### A.2 Program (input.mlir)

```mlir
module {

  func.func @matmul_impl(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    // buffers
    // rows = ceil(shape * elem_bytes / row_width); spad row_width = 16B, acc row_width = 64B
    %spad_a0 = perf.alloc : memref<512xi8, #perf.space<"spad", 0, 32>>
    %spad_b  = perf.alloc : memref<512xi8, #perf.space<"spad", 32, 32>>
    %acc_c0  = perf.alloc : memref<256xi32, #perf.space<"acc", 0, 16>>

    // DMA load A and B into spad
    %la = perf.dma_load %A, %spad_a0 [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %spad_b [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    // array init: synthesize zeros into PE accumulators (no SRAM read)
    %p0 = perf.array_init {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : () -> !perf.token

    // compute
    %c0 = perf.compute %spad_a0, %spad_b, %acc_c0 [%la, %lb, %p0] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    // store result
    %st0 = perf.dma_store %acc_c0, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.fence [%st0]
    return
  }
}
```
