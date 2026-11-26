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

### 2.6 `perf.token.init`

Produces an initial token and takes no operands. This op and loop-carried tokens must not be removed by DCE.

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

## 5. Hardware Configuration

The configuration describes structure only and supplies no complete cycle formula.

### 5.1 The Two Configuration Layers

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

### 5.2 Deriving the Contract

The vocabulary of executors and links is not written by hand. `TargetModel::build` projects it from the design point and materializes it as `perf.contract_def`, observable through `perf --dump-contract`. Program IR refers only to `#perf.res<kind, index>`.

Bank ports work the same way: the engine opens a ledger for each bank's read and write port from the memspace declaration. They may appear in Demands and in reports, and they may not be referenced by IR placement.

### 5.3 Arbitration Priority

The role names in the arbitration priority table, namely `execute`, `dma_load`, `dma_store` and `zero_writer`, are client classes at the level of the target contract. When the cost model quotes a Demand it assigns the op kind to a role, and the engine judges by the table.

One bank mapping is currently defined, `high_bits`, meaning `bank = row / rows_per_bank`.

---

## 6. Acceptance Criteria

Acceptance has three levels:

1. **Per-module error.** Compare DMA, array, bank arbitration, bubbles and writeback against the reference implementation separately, rather than comparing only total cycles.
2. **Error across a stated coverage range.** Report mean and maximum error within a stated range of workloads, dataflows, array sizes and tile sizes.
3. **Ranking confidence.** Promise that a ranking is reliable only when the predicted difference between two schedules exceeds the calibration uncertainty.

The target is no more than 10% error on key scenarios, and it must always be stated together with its coverage range. The specification does not require that any two schedules be ranked consistently.

---

## 7. Supported Scope

### 7.1 IR

`perf.alloc`, `perf.token.init`, `perf.dma_load`, `perf.dma_store`, `perf.array_init`, `perf.compute`, `perf.fence`.

### 7.2 Hardware

One systolic array; a fixed OS dataflow; a banked scratchpad and accumulator, with row-address intervals, bank mapping and priority arbitration on ports; one DMA read channel; one DMA write channel; fixed DRAM latency and bandwidth.

### 7.3 Simulation

A single completion token; mutual exclusion expressed through phase chains and the four demand landing modes; interleaved declaration at phase granularity; the `start`, `free_at` and `done` timestamps; per-instance event-driven replay; cycle-by-cycle priority arbitration on bank ports with exact fast-forwarding.

---

## 8. Gemmini Correlation Study

### 8.1 Role of Each Reference Tool

| Tool | Use |
|---|---|
| Spike | Functional correctness only; never a performance reference |
| Verilator / VCS | Primary cycle-accurate calibration source |
| FireSim | Cycle-accurate comparison for larger end-to-end workloads |
| Genus / Innovus | Area, frequency and power data; outside this simulator's scope |

Gemmini's storage contract, arbitration rules and compute access-stream generation are documented in `gemmini-target-profile.md`, which is authoritative for configuration instances.

### 8.2 The Correlation Loop

1. Pick a kernel, starting from a 64×64×64 matmul.
2. Generate IR and a schedule with this compiler, run this simulator, and obtain `C_sim`.
3. Construct an equivalent computation on Gemmini, run Verilator, VCS or FireSim, and obtain `C_rtl`.
4. Compare DMA, array, bank arbitration, bubbles and writeback separately.
5. Correct EstimatedTiming and the configuration.
6. Report the error within a stated coverage range.

There are two ways to construct the equivalent computation. The end-to-end route has this compiler emit Gemmini RoCC code and run it under a cycle-accurate simulator; it is the strictest and the most work. The hand-written route writes Gemmini C code whose tile, DMA and compute order mirrors the schedule in this IR, which suits quick calibration.

### 8.3 Order of Error Attribution

1. DMA fixed latency and bandwidth.
2. SRAM row request streams and bank/port arbitration, including A/B/C priority on one bank and DMA against the array.
3. Byte counts and ports of `array_init` and writeback.
4. Array stream and initiation interval, and the `first_mac`/`last_mac` cadence.
5. Transposer and OS row-order corrections.
6. Cross-resource bubbles.

---

## 9. Rationale for a Native MLIR Implementation

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
