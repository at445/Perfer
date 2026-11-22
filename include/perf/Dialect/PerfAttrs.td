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
  let mnemonic = "executor_family";
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
    "bool":$in_order,
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
  }];
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "int64_t":$count
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
    "bool":$inter_tile_pipelined
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_MemSpaceAttr : Perf_AttrDef<"MemSpace"> {
  let mnemonic = "memspace";
  let summary = "Named on-chip memory space: capacity, banking, ports and row geometry.";
  let parameters = (ins
    StringRefParameter<"space name referenced by #perf.space">:$name,
    "int64_t":$capacity_kib,
    "int64_t":$banks,
    "int64_t":$read_ports,
    "int64_t":$write_ports,
    "int64_t":$port_width_bytes,
    "int64_t":$row_width_bytes
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_DmaAttr : Perf_AttrDef<"Dma"> {
  let mnemonic = "dma";
  let parameters = (ins
    "int64_t":$bus_bytes_per_cycle,
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

def Perf_DesignAttr : Perf_AttrDef<"Design"> {
  let mnemonic = "design";
  let summary = "DSE-visible hardware configuration: physical structure parameters.";
  let description = [{
    The design point under evaluation, expressed in the DSE engineer's
    vocabulary: memory-space geometry, port counts, and subsystem
    microarchitecture packages. The compiler-simulator contract
    (executor instances, links) is projected from these physical
    parameters by TargetModel::build and materialized as a #perf.contract
    for inspection; users never write executor kinds by hand.
  }];
  let parameters = (ins
    "int64_t":$version,
    StringRefParameter<"design point name">:$target_name,
    "int64_t":$frequency_mhz,
    "::mlir::ArrayAttr":$memory_spaces,
    "int64_t":$dma_read_ports,
    "int64_t":$dma_write_ports,
    "int64_t":$arrays,
    "::perf::DmaAttr":$dma_channel,
    "::perf::ArrayAttr":$array_uarch,
    "bool":$share_dram_bus
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Perf_ContractAttr : Perf_AttrDef<"Contract"> {
  let mnemonic = "contract";
  let summary = "Materialized compiler-simulator contract: executor instances and links.";
  let description = [{
    The executor-level view projected from a #perf.design. This is the
    vocabulary program IR references via #perf.res and the vocabulary the
    simulator's resource ledgers are keyed on. It is generated by
    TargetModel::build and printed by --dump-contract for traceability;
    it is never written by hand.
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

def Perf_TargetAttr : Perf_AttrDef<"Target"> {
  let mnemonic = "target";
  let summary = "Complete hardware configuration consumed by the simulator.";
  let description = [{
    Legacy combined form (design + prior in one attribute). Retained for
    backward compatibility; new configs should use #perf.design and
    #perf.prior separately.
  }];
  let parameters = (ins
    "int64_t":$version,
    StringRefParameter<"target name">:$target_name,
    "int64_t":$frequency_mhz,
    "::mlir::ArrayAttr":$memory_spaces,
    "::mlir::ArrayAttr":$executors,
    "::mlir::ArrayAttr":$links,
    "::perf::CalibrationAttr":$calibration
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // PERF_ATTRS_TD
