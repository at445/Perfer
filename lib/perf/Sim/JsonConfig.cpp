// JsonConfig.cpp - Translate architect-level config.json into a TargetAttr.

#include "perf/Sim/JsonConfig.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace perf;

namespace {

// Helpers to extract required fields from a json::Object with error reporting.
struct JsonReader {
  const json::Object &obj;
  StringRef what;  // human-readable context for error messages
  bool ok = true;

  explicit JsonReader(const json::Object &obj, StringRef what)
      : obj(obj), what(what) {}

  int64_t getInt(StringRef key) {
    if (auto v = obj.getInteger(key))
      return *v;
    return fail(key), 0;
  }

  bool getBool(StringRef key) {
    if (auto v = obj.getBoolean(key))
      return *v;
    return fail(key), false;
  }

  StringRef getStr(StringRef key) {
    if (auto v = obj.getString(key))
      return *v;
    return fail(key), "";
  }

  // Optional string field with a fallback default (never fails).
  std::string getStrOr(StringRef key, StringRef fallback) {
    if (auto v = obj.getString(key))
      return v->str();
    return fallback.str();
  }

  const json::Array *getArr(StringRef key) {
    if (auto *v = obj.getArray(key))
      return v;
    return fail(key), nullptr;
  }

  const json::Object *getObj(StringRef key) {
    if (auto *v = obj.getObject(key))
      return v;
    return fail(key), nullptr;
  }

  void fail(StringRef key) {
    errs() << "error: config.json: '" << what << "' missing or invalid field '"
           << key << "'\n";
    ok = false;
  }
};

} // namespace

FailureOr<perf::TargetAttr>
perf::targetAttrFromJsonConfig(StringRef filename,
                               mlir::MLIRContext &context) {
  auto bufferOr = MemoryBuffer::getFile(filename);
  if (!bufferOr) {
    errs() << "error: cannot open config file '" << filename << "'\n";
    return failure();
  }
  auto jsonOr = json::parse((*bufferOr)->getBuffer());
  if (!jsonOr) {
    errs() << "error: failed to parse JSON in '" << filename
           << "': " << toString(jsonOr.takeError()) << "\n";
    return failure();
  }
  auto *root = jsonOr->getAsObject();
  if (!root) {
    errs() << "error: config.json root is not an object\n";
    return failure();
  }

  JsonReader r(*root, "root");
  int64_t version = r.getInt("version");
  std::string targetName = r.getStr("target_name").str();
  int64_t frequencyMhz = r.getInt("frequency_mhz");
  const json::Array *entities = r.getArr("entities");
  if (!r.ok)
    return failure();

  auto ctx = &context;

  // ── pass 1: on-chip SRAM spaces (their names give every executor's
  // reachable-space set in this flat connectivity model) ─────────────────
  SmallVector<mlir::Attribute> spaceAttrs;
  SmallVector<mlir::Attribute> allSpaceNames;
  for (const json::Value &v : *entities) {
    const json::Object *e = v.getAsObject();
    if (!e || e->getString("type") != "sram")
      continue;
    std::string id = e->getString("id").value_or("").str();
    const json::Object *attrs = e->getObject("attrs");
    if (!attrs) {
      errs() << "error: config.json: sram entity '" << id << "' missing attrs\n";
      return failure();
    }
    JsonReader ar(*attrs, id);
    spaceAttrs.push_back(perf::MemSpaceAttr::get(
        ctx, id, ar.getInt("capacity_kib"), ar.getInt("banks"),
        ar.getInt("read_ports"), ar.getInt("write_ports"),
        ar.getInt("port_width_bytes"), ar.getInt("row_width_bytes"),
        ar.getStrOr("interleave", "low")));
    if (!ar.ok)
      return failure();
    allSpaceNames.push_back(mlir::StringAttr::get(ctx, id));
  }
  auto reachability = mlir::ArrayAttr::get(ctx, allSpaceNames);

  // ── pass 2: dram -> dram_bus link; array / dma -> executor families ────
  SmallVector<mlir::Attribute> linkAttrs;
  SmallVector<mlir::Attribute> execAttrs;
  bool sawDram = false, sawArray = false, sawDma = false;

  for (const json::Value &v : *entities) {
    const json::Object *e = v.getAsObject();
    if (!e)
      continue;
    std::string type = e->getString("type").value_or("").str();
    std::string id = e->getString("id").value_or("").str();
    const json::Object *attrs = e->getObject("attrs");
    if (!attrs) {
      if (type == "dram" || type == "array" || type == "dma") {
        errs() << "error: config.json: entity '" << id << "' missing attrs\n";
        return failure();
      }
      continue;
    }
    JsonReader ar(*attrs, id);

    if (type == "dram") {
      // Off-chip DRAM channels become the dram_bus link (the bandwidth
      // source for DMA transfer nominal).
      int64_t channels = ar.getInt("channels");
      int64_t bpc = ar.getInt("bytes_per_cycle");
      if (!ar.ok)
        return failure();
      linkAttrs.push_back(perf::LinkAttr::get(ctx, "dram_bus", channels, bpc));
      sawDram = true;
    } else if (type == "array") {
      auto arrayUarch = perf::ArrayAttr::get(
          ctx, ar.getStr("topology"), ar.getInt("mesh_rows"),
          ar.getInt("mesh_cols"), ar.getInt("tile_rows"),
          ar.getInt("tile_cols"), ar.getInt("pe_latency_cycles"),
          ar.getBool("input_skew"), ar.getBool("inter_tile_pipelined"),
          ar.getInt("acc_read_scale_cycles"),
          ar.getInt("acc_read_act_cycles"));
      if (!ar.ok)
        return failure();
      execAttrs.push_back(perf::ExecutorFamilyAttr::get(
          ctx, "array", /*count=*/1, reachability, arrayUarch));
      sawArray = true;
    } else if (type == "dma") {
      // A DMA engine's read/write port counts expand into the dma_read /
      // dma_write executor families, both sharing the engine config.
      auto dmaCfg = perf::DmaAttr::get(
          ctx, ar.getInt("read_ports"), ar.getInt("write_ports"),
          ar.getInt("fixed_latency_cycles"), ar.getInt("max_outstanding"),
          ar.getInt("burst_bytes"));
      int64_t readPorts = ar.getInt("read_ports");
      int64_t writePorts = ar.getInt("write_ports");
      if (!ar.ok)
        return failure();
      execAttrs.push_back(perf::ExecutorFamilyAttr::get(
          ctx, "dma_read", readPorts, reachability, dmaCfg));
      execAttrs.push_back(perf::ExecutorFamilyAttr::get(
          ctx, "dma_write", writePorts, reachability, dmaCfg));
      sawDma = true;
    }
    // unknown entity types are ignored (forward compatibility)
  }

  if (!sawDram || !sawArray || !sawDma) {
    errs() << "error: config.json: required entity types missing (need dram, "
              "array, dma)\n";
    return failure();
  }

  return perf::TargetAttr::get(ctx, version, targetName, frequencyMhz,
                               mlir::ArrayAttr::get(ctx, spaceAttrs),
                               mlir::ArrayAttr::get(ctx, execAttrs),
                               mlir::ArrayAttr::get(ctx, linkAttrs));
}
