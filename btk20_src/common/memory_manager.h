/**
 * @file memory_manager.h
 * @brief A simple memory manager
 * @author John McDonough and Kenichi Kumatani
 */
#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <list>
#include <typeinfo>

#include "common/mlist.h"

//#define  DEBUG_MEMORY_MANAGER

// ----- definition of class `MemoryAllocator' -----
//
class MemoryAllocator {
  union Element_ {
    Element_* _next;
  };

  typedef list<char*>          AllocList_;
  typedef AllocList_::iterator AllocListIterator_;

 public:
  MemoryAllocator(unsigned elem_size, unsigned blk_size = 1000, unsigned limit = 0);
  ~MemoryAllocator();

  void* new_elem();
  void  deleteElem(void* e);

  void report(FILE* fp = stdout) const;

  const unsigned cnt()       const { return cnt_;     };
  const unsigned block_size() const { return blk_size_; };
  const size_t   size()      const { return size_;    };
  void  set_limit(unsigned limit) { limit_ = limit; }

 private:
  void new_block_();

  const size_t					size_;
  const unsigned				blk_size_;

  AllocList_					alloc_list_;
  unsigned					cnt_;
  unsigned					limit_;
#ifdef DEBUG_MEMORY_MANAGER
  Element_*					list_;
#endif
};


// ----- definition of class `BasicMemoryManager' -----
//
class BasicMemoryManager {

  typedef map<unsigned, MemoryAllocator*>	AllocatorList_;
  typedef AllocatorList_::iterator		AllocatorListIterator_;
  typedef AllocatorList_::value_type		AllocatorList_ValueType;

 public:
  BasicMemoryManager(unsigned elem_size, unsigned blk_size = 1000, unsigned limit = 0);
  ~BasicMemoryManager();

  inline void* new_elem_() { return allocator_->new_elem(); }
  inline void  delete_elem_(void* e) { allocator_->deleteElem(e); }

  void report(FILE* fp = stdout) { allocator_->report(fp); }

  const unsigned cnt()       const { return allocator_->cnt();       };
  const unsigned block_size() const { return allocator_->block_size(); };
  const size_t   size()      const { return allocator_->size();      };
  void set_limit(unsigned limit)    { allocator_->set_limit(limit); }

 private:
  MemoryAllocator* initialize_(unsigned elem_size, unsigned blk_size, unsigned limit);

  static AllocatorList_				allocator_list_;
  MemoryAllocator*				allocator_;
};


// ----- definition of class template `MemoryManager' -----
//
template <class Type>
class MemoryManager : public BasicMemoryManager {
 public:
  MemoryManager(const String& type, unsigned blk_size = 1000, unsigned limit = 0);
  ~MemoryManager();

  Type* new_elem() { return (Type*) new_elem_(); }
  void  delete_elem(void* e) { delete_elem_(e); }

  const String& type() const { return type_; }

 private:
  const String					type_;
};



// ----- methods for class template `MemoryManager' -----
//
template <class Type>
MemoryManager<Type>::
MemoryManager(const String& type, unsigned blk_size, unsigned limit)
  : BasicMemoryManager(sizeof(Type), blk_size, limit), type_(type)
{
  cout << "Creating 'MemoryManager' for type '" << type << "'." << endl;
}

template <class Type>
MemoryManager<Type>::~MemoryManager() { }

#endif
