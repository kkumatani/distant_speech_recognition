/**
 * @file refcount.h
 * @brief reference counter
 * @author Fabian Jakobs
 */
#include "common/refcount.h"

// can this be included directly in 'refcount.h'?
MemoryManager<ReferenceCount::ReferenceCount_>& ReferenceCount::ReferenceCount_::memoryManager() {
  static MemoryManager<ReferenceCount::ReferenceCount_> memoryManager_("ReferenceCount::ReferenceCount_");
  return memoryManager_;
}
