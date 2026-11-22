// PerfOps.td — perf dialect MVP op definitions

#ifndef PERF_OPS_TD
#define PERF_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "perf/Dialect/PerfDialect.td"
include "perf/Dialect/PerfAttrs.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def Perf_TokenType : TypeDef<PerfDialect, "Token"> {
  let mnemonic = "token";
  let summary = "Happens-before token between side-effecting ops.";
  let genStorageClass = 1;
  let builders = [TypeBuilder<(ins), [{
    return $_get($_ctxt);
  }]>];
}

def Perf_TokenOperand : Type<CPred<"::llvm::isa<::perf::TokenType>($_self)">,
                             "token dependency",
                             "::perf::TokenType"> {
  let builderCall = "$_builder.getType<::perf::TokenType>()";
}

//===----------------------------------------------------------------------===//
// Op base class
//===----------------------------------------------------------------------===//

class Perf_Op<string mnemonic, list<Trait> traits = []> :
    Op<PerfDialect, mnemonic, traits>;

// MemoryEffectOpInterface method declaration helper.
class Perf_MemEffectOp<string mnemonic, list<Trait> traits = []> :
    Perf_Op<mnemonic, traits # [MemoryEffectsOpInterface]> {
  let extraClassDeclaration = [{
    void getEffects(::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
            &effects);
  }];
}

//===----------------------------------------------------------------------===//
// perf.target_def
//===----------------------------------------------------------------------===//

def Perf_TargetDefOp : Perf_Op<"target_def", [Symbol, NoMemoryEffect]> {
  let summary = "Top-level hardware design-point definition (DSE-visible config).";
  let description = [{
    A top-level declaration of a hardware design point: the parameters a
    DSE sweep varies (memory spaces, executor families, links, frequency).
    Simulation priors live in a separate perf.prior_def so sweeps never
    fork the prior state.

    Example:
      perf.target_def @systolic_npu_v1 #perf.design<...>
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Perf_DesignAttr:$config
  );
  let assemblyFormat = "$sym_name $config attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.prior_def
//===----------------------------------------------------------------------===//

def Perf_PriorDefOp : Perf_Op<"prior_def", [Symbol, NoMemoryEffect]> {
  let summary = "Top-level simulation-prior definition (calibration state).";
  let description = [{
    A top-level declaration of simulation priors: calibration corrections
    produced by alignment against RTL ground truth. Shared unchanged
    across design points of the same machine family.

    Example:
      perf.prior_def @gemmini_base #perf.prior<...>
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Perf_PriorAttr:$config
  );
  let assemblyFormat = "$sym_name $config attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.contract_def
//===----------------------------------------------------------------------===//

def Perf_ContractDefOp : Perf_Op<"contract_def", [Symbol, NoMemoryEffect]> {
  let summary = "Materialized compiler-simulator contract (generated, never hand-written).";
  let description = [{
    The executor-level contract projected from a perf.target_def design
    point. Emitted by `perf --dump-contract` so that both the compiler
    (which must place #perf.res references) and the simulator consume the
    same explicit vocabulary.

    Example:
      perf.contract_def @systolic_npu_v1 #perf.contract<...>
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Perf_ContractAttr:$config
  );
  let assemblyFormat = "$sym_name $config attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.alloc
//===----------------------------------------------------------------------===//

def Perf_AllocOp : Perf_Op<"alloc", [NoMemoryEffect]> {
  let summary = "Compiler-placed static buffer in a #perf.space memory space.";
  let description = [{
    Produces a root buffer: a memref with static shape and a #perf.space
    memory space. Root buffers are the roots of the provenance analysis.
  }];
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.token.init
//===----------------------------------------------------------------------===//

def Perf_TokenInitOp : Perf_Op<"token.init", [NoMemoryEffect]> {
  let summary = "Initial free token for a buffer rotation group.";
  let description = [{
    Creates an unconstrained token used as the starting point of a
    token-carry chain (e.g. buffer rotation via iter_args).
  }];
  let results = (outs Perf_TokenType:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// perf.dma_load / perf.dma_store
//===----------------------------------------------------------------------===//

def Perf_DmaLoadOp : Perf_MemEffectOp<"dma_load"> {
  let summary = "Async DMA from main memory to an on-chip buffer.";
  let description = [{
    Explicit DMA load from a main-memory memref to an on-chip buffer.
    Direction is fixed by the op name. Placement is given by resource.
  }];
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Variadic<Perf_TokenOperand>:$deps,
    I64Attr:$bytes,
    Perf_ResRefAttr:$resource
  );
  let results = (outs Perf_TokenType:$token);
  let assemblyFormat = [{
    $src `,` $dst (`[` $deps^ `]`)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  // C++ getEffects: Read on src, Write on dst.
}

def Perf_DmaStoreOp : Perf_MemEffectOp<"dma_store"> {
  let summary = "Async DMA from an on-chip buffer to main memory.";
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Variadic<Perf_TokenOperand>:$deps,
    I64Attr:$bytes,
    Perf_ResRefAttr:$resource
  );
  let results = (outs Perf_TokenType:$token);
  let assemblyFormat = [{
    $src `,` $dst (`[` $deps^ `]`)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  // C++ getEffects: Read on src, Write on dst.
}

//===----------------------------------------------------------------------===//
// perf.array_init
//===----------------------------------------------------------------------===//

def Perf_ArrayInitOp : Perf_MemEffectOp<"array_init",
    [AttrSizedOperandSegments]> {
  let summary = "Initialize the array's PE accumulator register state.";
  let description = [{
    Establishes the initial state of the PE accumulator registers.
    The memref operand is Optional:

    - No operand: synthesizes zeros (zero-fill path). Does not read SRAM,
      declares no Read effect, and produces no RAW edge.
    - With operand: reads the old partial sums / bias in that region into
      the PE registers, producing an acc bank read.
  }];
  let arguments = (ins
    Optional<AnyMemRef>:$dst,
    Variadic<Perf_TokenOperand>:$deps,
    I64Attr:$m,
    I64Attr:$n,
    Perf_ResRefAttr:$resource
  );
  let results = (outs Perf_TokenType:$token);
  let assemblyFormat = [{
    ($dst^)? (`[` $deps^ `]`)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.compute
//===----------------------------------------------------------------------===//

def Perf_ComputeOp : Perf_MemEffectOp<"compute"> {
  let summary = "One array compute step of a tiled matmul.";
  let description = [{
    Always accumulates onto the array's current PE state (PE += A x B).
    The PE state must be established by a perf.array_init or a previous
    perf.compute, traceable along the token chain (R6). Compute never
    reads the C bank: C reads happen only in array_init (operand form),
    C writes only in the writeback phase.
  }];
  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$b,
    AnyMemRef:$c,
    Variadic<Perf_TokenOperand>:$deps,
    I64Attr:$m,
    I64Attr:$n,
    I64Attr:$k,
    BoolAttr:$writeback,
    Perf_ResRefAttr:$resource
  );
  let results = (outs Perf_TokenType:$token);
  let assemblyFormat = [{
    $a `,` $b `,` $c (`[` $deps^ `]`)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// perf.wait
//===----------------------------------------------------------------------===//

def Perf_WaitOp : Perf_Op<"wait", []> {
  let summary = "Join point for completion tokens.";
  let arguments = (ins Variadic<Perf_TokenOperand>:$tokens);
  let assemblyFormat = "(`[` $tokens^ `]`)? attr-dict";
}

#endif // PERF_OPS_TD
