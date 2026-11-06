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
  let summary = "On-chip memory space reference: (space name, bank index).";
  let parameters = (ins
    StringRefParameter<"space name: spad | acc | ...">:$name,
    "int64_t":$bank
  );
  let assemblyFormat = "`<` $name `,` $bank `>`";
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

//===----------------------------------------------------------------------===//
// Target config attributes
//===----------------------------------------------------------------------===//

def Perf_ResourceFamilyAttr : Perf_AttrDef<"ResourceFamily"> {
  let mnemonic = "resource_family";
  let summary = "Homogeneous family of schedulable/contention resources.";
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "int64_t":$count,
    "int64_t":$concurrency,
    "bool":$in_order,
    "bool":$placeable
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

def Perf_SramAttr : Perf_AttrDef<"Sram"> {
  let mnemonic = "sram";
  let parameters = (ins
    "int64_t":$capacity_kib,
    "int64_t":$banks,
    "int64_t":$read_ports,
    "int64_t":$write_ports,
    "int64_t":$port_width_bytes
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

def Perf_QueuesAttr : Perf_AttrDef<"Queues"> {
  let mnemonic = "queues";
  let parameters = (ins
    "int64_t":$load_queue_depth,
    "int64_t":$store_queue_depth,
    "int64_t":$execute_queue_depth,
    "int64_t":$dependency_entries
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
  let summary = "Complete hardware configuration consumed by the simulator.";
  let parameters = (ins
    "int64_t":$version,
    StringRefParameter<"target name">:$target_name,
    "int64_t":$frequency_mhz,
    "::perf::ArrayAttr":$array,
    "::perf::SramAttr":$scratchpad,
    "::perf::SramAttr":$accumulator,
    "::perf::DmaAttr":$dma,
    "::perf::QueuesAttr":$queues,
    "::mlir::ArrayAttr":$resources,
    "::perf::CalibrationAttr":$calibration
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // PERF_ATTRS_TD
