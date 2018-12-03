/**
 * @file memory_manager.cc
 * @brief A simple memory manager
 * @author John McDonough and Kenichi Kumatani
 */

#include "common/memory_manager.h"
#include <algorithm>

// ----- methods for class `MemoryAllocator' -----
//
MemoryAllocator::MemoryAllocator(unsigned elem_size, unsigned blk_size, unsigned limit)
  : size_(std::max<unsigned>(elem_size, sizeof(void*))), blk_size_(blk_size),
    cnt_(0), limit_(limit)
{
#ifdef DEBUG_MEMORY_MANAGER
  list_ = NULL;
#endif
  cout << "Creating allocator for objects of size " << elem_size << " ... " << endl;

  new_block_();
}

MemoryAllocator::~MemoryAllocator()
{
  /* this does not belong here! (merkosh) */
#ifdef DEBUG_MEMORY_MANAGER
  printf("Freeing all memory.\n");
#endif
  for (AllocListIterator_ itr = alloc_list_.begin(); itr != alloc_list_.end(); itr++)
    free(*itr);
}

void MemoryAllocator::new_block_()
{
#ifdef DEBUG_MEMORY_MANAGER
  if (limit_ > 0) {
    printf("Allocating block %lu with size %lu : Total Allocated %lu : Limit %u\n",
           alloc_list_.size(), size_ * blk_size_, alloc_list_.size() * size_ * blk_size_, limit_);
    fflush(stdout);
  }
#endif

  if (limit_ > 0 && (alloc_list_.size() * size_ * blk_size_) > limit_)
    throw jallocation_error("Tried to allocate more than %d bytes", limit_);

#ifdef DEBUG_MEMORY_MANAGER
  /* keep track of every pointer */
  char* list = (char*) malloc(size_ * blk_size_);
  alloc_list_.push_back(list);

  for (unsigned iblk = 0; iblk < blk_size_; iblk++) {
    Element_* elem = (Element_*) (list + (iblk * size_));
    elem->_next    = list_;
    list_          = elem;
  }
#endif
}

void MemoryAllocator::report(FILE* fp) const
{
  fprintf(fp, "\nMemory Manager\n");
  fprintf(fp, "C++ Type:  %s\n", typeid(this).name());
  fprintf(fp, "Type Size: %lu\n", size_);
  fprintf(fp, "Space allocated for %lu objects\n",
          alloc_list_.size() * blk_size_);
  fprintf(fp, "Total allocated space is %lu\n",
          alloc_list_.size() * blk_size_ * size_);
  fprintf(fp, "There are %d allocated objects\n", cnt_);
  fprintf(fp, "\n");

  fflush(fp);
}

void* MemoryAllocator::new_elem()
{
  cnt_++;
#ifdef DEBUG_MEMORY_MANAGER
  printf("Allocating element %d\n", cnt_);
  if (list_ == NULL)
    new_block_();

  Element_* e = list_;
  list_ = e->_next;
  return e;
#else /* DEBUG_MEMORY_MANAGER */
  return (void *) malloc(size_);
#endif
}

void MemoryAllocator::deleteElem(void* e)
{
#ifdef DEBUG_MEMORY_MANAGER
  printf("Deleting element %d\n", cnt_);
#endif

  if (e == NULL) return;

  cnt_--;

#ifdef DEBUG_MEMORY_MANAGER
  Element_* elem = (Element_*) e;
  elem->_next = list_;
  list_ = elem;
#else
  free(e);
#endif
}

// ----- methods for class `BasicMemoryManager' -----
//
BasicMemoryManager::AllocatorList_ BasicMemoryManager::allocator_list_;

BasicMemoryManager::BasicMemoryManager(unsigned elem_size, unsigned blk_size, unsigned limit)
  : allocator_(initialize_(elem_size, blk_size, limit)) { }

BasicMemoryManager::~BasicMemoryManager() { }

MemoryAllocator* BasicMemoryManager::initialize_(unsigned elem_size, unsigned blk_size, unsigned limit)
{
  AllocatorListIterator_ itr = allocator_list_.find(elem_size);
  if (itr == allocator_list_.end()) {
    allocator_list_.insert(AllocatorList_ValueType(elem_size, new MemoryAllocator(elem_size, blk_size, limit)));
    itr = allocator_list_.find(elem_size);
  }

  return (*itr).second;
}
