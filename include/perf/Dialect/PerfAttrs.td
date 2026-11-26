// PerfAttrs.td

#ifndef PERF_ATTRS_TD
#define PERF_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "perf/Dialect/PerfDialect.td"

class Perf_AttrDef<string name> : AttrDef<PerfDialect, name>;

//===----------------------------------------------------------------------===//
// Execution IR references
//===----------------------------------------------------------------------===//

def Perf_SpaceAttr : Perf_AttrDef<"Space"> {
  let mnemonic = "space";
  let summary = "On-chip memory space reference: (space name, base_row, rows).";
  let parameters = (ins
    StringRefParameter<"space name: spad | acc | ...">:$name,
    "int64_t":$base_row,
    "int64_t":$rows
  );
  let assemblyFormat = "`<` $name `,` $base_row `,` $rows `>`";
}

def Perf_ResRefAttr : Perf_AttrDef<"ResRef"> {
  let mnemonic = "res";
  let summary = "Resource reference: (kind, index) into a resource family declared by the target config.";
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "int64_t":$index
  );
  let assemblyFormat = "`<` $kind `,` $index `>`";
}

def Perf_ScheduleEntryAttr : Perf_AttrDef<"ScheduleEntry"> {
  let mnemonic = "schedule_entry";
  let summary = "Total order of schedulable ops on one resource instance.";
  let parameters = (ins
    "::perf::ResRefAttr":$resource,
    "::mlir::ArrayAttr":$ops
  );
  let assemblyFormat = "`<` $resource `,` $ops `>`";
}

//===----------------------------------------------------------------------===//
// Target config attributes
//===----------------------------------------------------------------------===//

def Perf_ExecutorFamilyAttr : Perf_AttrDef<"ExecutorFamily"> {
  let mnemonic = "executor";
  let summary = "Homogeneous family of executor resources referenced by IR placement.";
  let description = [{
    Declares a family of executor instances: DMA channels, arrays, etc.
    Executors are the only resources that may be referenced by an op's
    #perf.res placement. Mutual exclusion is expressed by phase-chain
    placement on the resource ledger, not by a family-level concurrency
    slot count. DMA in-flight limits are carried by the family's own
    config (#perf.dma::max_outstanding), not here.

    The config attribute carries the family's microarchitectural
    parameters: #perf.dma for DMA families, #perf.array for array
    families. The kind-to-config-type correspondence is validated when
    the target is built.
  }];
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "int64_t":$count,
    "::mlir::ArrayAttr":$spaces,
    "::mlir::Attribute":$config
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_LinkAttr : Perf_AttrDef<"Link"> {
  let mnemonic = "link";
  let summary = "Internal contention channel referenced only by cost-model phase chains.";
  let description = [{
    Declares an internal contention resource (e.g. dram_bus, acc write port
    as a shared medium). Links may not be referenced by IR placement; they
    appear only in the demands of an EstimatedTiming phase chain.

    bytes_per_cycle is the per-channel transfer bandwidth: the simulator
    derives transfer nominal (bytes / bandwidth) from it. How contention on
    the channel resolves (fifo for dram_bus) is a fixed rule built into the
    simulator, keyed off the link kind, not carried here.
  }];
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "int64_t":$count,
    "int64_t":$bytes_per_cycle
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_ArrayAttr : Perf_AttrDef<"Array"> {
  let mnemonic = "array";
  let parameters = (ins
    StringRefParameter<"array topology">:$topology,
    "int64_t":$mesh_rows,
    "int64_t":$mesh_cols,
    "int64_t":$tile_rows,
    "int64_t":$tile_cols,
    "int64_t":$pe_latency_cycles,
    "bool":$input_skew,
    "bool":$inter_tile_pipelined,
    // Gemmini acc read-out epilogue: int32 partial sums are scaled (x f32)
    // and activated (ReLU) while leaving the accumulator. Pipeline stages
    // on the acc read port, charged on a down-converting acc read.
    "int64_t":$acc_read_scale_cycles,
    "int64_t":$acc_read_act_cycles
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_MemSpaceAttr : Perf_AttrDef<"MemSpace"> {
  let mnemonic = "sram";
  let summary = "Named on-chip memory space: capacity, banking, ports, row geometry, and bank interleave.";
  let description = [{
    The geometry of one on-chip SRAM space. `interleave` is the row-to-bank
    mapping policy used by the simulator to expand a #perf.space access into
    per-bank demands:
      "low"   bank = row % banks   (consecutive rows spread across banks)
      "high"  bank = row / rowsPerBank (consecutive rows packed in one bank)
    Per-bank contention resolution is a fixed rule built into the simulator
    (per-cycle priority), not carried here.
  }];
  let parameters = (ins
    StringRefParameter<"space name referenced by #perf.space">:$name,
    "int64_t":$capacity_kib,
    "int64_t":$banks,
    "int64_t":$read_ports,
    "int64_t":$write_ports,
    "int64_t":$port_width_bytes,
    "int64_t":$row_width_bytes,
    StringRefParameter<"row-to-bank interleave policy: low | high">:$interleave
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_DmaAttr : Perf_AttrDef<"Dma"> {
  let mnemonic = "dma";
  let summary = "DMA engine parameters, including its read/write port counts.";
  let description = [{
    Engine-intrinsic DMA parameters plus the engine's read/write port counts
    (which a json->target translation expands into the dma_read / dma_write
    executors). The transfer bandwidth is NOT carried here: it belongs to the
    shared DRAM channel (the dram_bus link's bytes_per_cycle), the source of
    DMA transfer nominal (fixed_latency_cycles + bytes / link bandwidth).
  }];
  let parameters = (ins
    "int64_t":$read_ports,
    "int64_t":$write_ports,
    "int64_t":$fixed_latency_cycles,
    "int64_t":$max_outstanding,
    "int64_t":$burst_bytes
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_CalibrationAttr : Perf_AttrDef<"Calibration"> {
  let mnemonic = "calibration";
  let parameters = (ins
    "int64_t":$extra_fill_cycles,
    "int64_t":$extra_drain_cycles,
    "int64_t":$extra_preload_cycles,
    "int64_t":$extra_writeback_cycles,
    "int64_t":$host_issue_cycles
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_TargetAttr : Perf_AttrDef<"Target"> {
  let mnemonic = "target";
  let summary = "Simulator-level hardware configuration: the executor/link vocabulary.";
  let description = [{
    The complete, self-contained hardware configuration the simulator
    consumes directly: the memory spaces, executor families, and links, with
    their physical parameters. This is the single vocabulary that program IR
    references via #perf.res and that the simulator's resource ledgers are
    keyed on. It is produced from an architect-level config.json by
    `perf --config ... --dump-target`, and read back by `--target`.

    How each resource is modelled (the demand type, placement strategy,
    arbiter, and nominal cycle formula) is keyed off the executor/link kind
    and built into the simulator; it is not carried here.
  }];
  let parameters = (ins
    "int64_t":$version,
    StringRefParameter<"design point name">:$target_name,
    "int64_t":$frequency_mhz,
    "::mlir::ArrayAttr":$memory_spaces,
    "::mlir::ArrayAttr":$executors,
    "::mlir::ArrayAttr":$links
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_PriorAttr : Perf_AttrDef<"Prior"> {
  let mnemonic = "prior";
  let summary = "Simulation priors: calibration corrections shared across design points.";
  let description = [{
    Prior knowledge about the machine, produced by calibration against
    RTL ground truth. Shared unchanged across design points of the same
    machine family; a DSE sweep must never mutate it per-candidate.
  }];
  let parameters = (ins
    "int64_t":$version,
    StringRefParameter<"prior family name">:$target_name,
    "::perf::CalibrationAttr":$calibration
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // PERF_ATTRS_TD
