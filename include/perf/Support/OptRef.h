// OptRef.h - Optional non-owning reference wrapper.

#ifndef PERF_SUPPORT_OPTREF_H
#define PERF_SUPPORT_OPTREF_H

#include <functional>
#include <optional>

namespace perf {

// Optional non-owning reference, in the spirit of llvm::optional but for
// references (C++26 std::optional<T&>). Lets lookup APIs signal "not found"
// without a nullptr sentinel. Unlike std::optional<reference_wrapper<T>>,
// operator-> / operator* forward straight to the referenced object.
template <typename T> class OptRef {
public:
  OptRef() = default;
  OptRef(std::nullopt_t) {}
  OptRef(T &value) : ptr_(&value) {}
  OptRef(std::reference_wrapper<T> ref) : ptr_(&ref.get()) {}

  bool has_value() const { return ptr_ != nullptr; }
  explicit operator bool() const { return has_value(); }

  T &value() const { return *ptr_; }
  T &operator*() const { return *ptr_; }
  T *operator->() const { return ptr_; }

private:
  T *ptr_ = nullptr;
};

template <typename T> using OptCRef = OptRef<const T>;

} // namespace perf

#endif // PERF_SUPPORT_OPTREF_H
