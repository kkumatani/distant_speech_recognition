/**
 * @file refcount.h
 * @brief Implementing reference counter
 * @author Fabian Jakobs
 */

#ifndef REFCOUNT_H
#define REFCOUNT_H

#include "common/memory_manager.h"

/**
 *  @class template to assist with upcasting
 */
template <class To, class From>
To& Cast(const From& from) { return *((To *) &from); }

/**
 * @class class template to give smart pointers the same inheritance as the object pointed to
 */
template <class DerivedType, class BaseTypePtr>
class Inherit : public BaseTypePtr {
 public:
  Inherit(DerivedType* s = NULL) : BaseTypePtr(s) { }

  DerivedType* operator->() const { return (DerivedType*) BaseTypePtr::the_p; }

  /*
        DerivedType& operator*() { return *((DerivedType*) the_p); }
  const DerivedType& operator*() { return *((DerivedType*) the_p); }
  */
};

class ReferenceCount {
  template<class T>
    friend class refcount_ptr;
  class ReferenceCount_ {
  public:
    ReferenceCount_(unsigned c)
      : count_(c) { }

    void* operator new(size_t sz) { return memoryManager().new_elem(); }
    void operator delete(void* e) { memoryManager().delete_elem(e); }

    static MemoryManager<ReferenceCount_>& memoryManager();

    unsigned		count_;
  };

 public:
  // create with count of 1
  ReferenceCount(): p_refcnt(new ReferenceCount_(1)) { };
  // copy and increment count
  ReferenceCount(const ReferenceCount& anRC): p_refcnt(anRC.p_refcnt) {
    p_refcnt->count_++;
  };

  // decrement count, delete if 0
  ~ReferenceCount() { decrement(); }

  // Assign, decrement lhs count, increment rhs
  ReferenceCount& operator=(const ReferenceCount& rhs) {
    rhs.p_refcnt->count_++;
    decrement();
    p_refcnt = rhs.p_refcnt;
    return *this;
  }

  // True if count is 1
  bool unique() const { return p_refcnt->count_ == 1;};

 private:
  ReferenceCount_* p_refcnt;

  // Decrement count; delete if 0
  void decrement() {
    if (unique()) delete p_refcnt;
    else p_refcnt->count_--;
  }
};


// Implementation of a reference-counted object pointer
// class as described in Barton and Nackman, 1988
template<class T>
class refcount_ptr {
 public:
  // construct pointing to a heap object
  refcount_ptr(T* newobj = NULL)
    : the_p(newobj), smart_behavior_(true) { }

  refcount_ptr(const refcount_ptr& rhs)
    : the_p(rhs.the_p), smart_behavior_(true), ref_cnt_p_(rhs.ref_cnt_p_) { }

  virtual ~refcount_ptr() {
    if (smart_behavior_) {
      if (unique()) delete the_p;
    } else {
      ref_cnt_p_.p_refcnt->count_++;
    }
  }

  void disable()
  {
    if (is_null())
      throw jconsistency_error("Attempted to disable a NULL pointer.");

    if (unique())
      throw jconsistency_error("Attempted to disable a unique pointer.");
    smart_behavior_ = false;
    ref_cnt_p_.p_refcnt->count_--;
  }

  refcount_ptr<T>& operator=(const refcount_ptr<T>& rhs) {
    if (the_p != rhs.the_p) {
      if (smart_behavior_) {
	if (unique()) delete the_p;
      } else {
	ref_cnt_p_.p_refcnt->count_++;
	smart_behavior_ = true;
      }
      the_p = rhs.the_p;
      ref_cnt_p_ = rhs.ref_cnt_p_;
    }
    return *this;
  }

  refcount_ptr<T>& operator=(T* rhs) {
    if (smart_behavior_) {
      if (unique()) delete the_p;
    } else {
      ref_cnt_p_.p_refcnt->count_++;
      smart_behavior_ = true;
    }
    the_p = rhs;
    ref_cnt_p_ = ReferenceCount();
    return *this;
  }

        T& operator*()        { return *the_p; }
  const T& operator*()  const { return *the_p; }

  T* operator->() const { return  the_p; }

  // Is count one?
  bool unique() const { return ref_cnt_p_.unique(); }

  // Is the_p pointing to NULL?
  bool is_null() const { return the_p == 0; }

  friend
    bool operator==(const refcount_ptr<T>& lhs, const refcount_ptr<T>& rhs) {
    return lhs.the_p == rhs.the_p;
  }
  friend
    bool operator!=(const refcount_ptr<T>& lhs, const refcount_ptr<T>& rhs) {
    return lhs.the_p != rhs.the_p;
  }

 protected:
  T*				the_p;

 private:
  bool 				smart_behavior_;
  ReferenceCount		ref_cnt_p_; // number of pointers to the heap object
};


// ----- definition of class `Countable' -----
//
class Countable {
 public:
  virtual ~Countable() { }

  bool unique() const { return count_ == 1; }
  void increment() { count_++;  assert(count_ < UINT_MAX); }
  void decrement() { assert(count_ > 0); count_--;  assert(count_ > 0); }

 protected:
  Countable() : count_(0) { }

 private:
  unsigned			count_;
};


// ----- definition of class template `refcountable_ptr' -----
//
template<class T>
class refcountable_ptr {
 public:
  // construct pointing to a heap object
  refcountable_ptr(T* newobj = NULL)
    : the_p(newobj), smart_behavior_(true) { increment(); }

  refcountable_ptr(const refcountable_ptr& rhs)
    : the_p(rhs.the_p), smart_behavior_(true) { increment(); }

  virtual ~refcountable_ptr() {
    if (smart_behavior_)
      decrement();
  }

  void disable()
  {
    if (is_null())
      throw jconsistency_error("Attempted to disable a NULL pointer.");

    if (unique())
      throw jconsistency_error("Attempted to disable a unique pointer.");
    smart_behavior_ = false;
    decrement();
  }

  refcountable_ptr<T>& operator=(const refcountable_ptr<T>& rhs) {
    if (the_p != rhs.the_p) {
      if (smart_behavior_)
	decrement();

      the_p = rhs.the_p;
      increment();
    }
    return *this;
  }

  refcountable_ptr<T>& operator=(T* rhs) {
    if (smart_behavior_)
      decrement();
    else
      smart_behavior_ = true;

    the_p = rhs;
    increment();

    return *this;
  }

        T& operator*()        { return Cast<T>(*the_p); }
  const T& operator*()  const { return Cast<T>(*the_p); }

        T* operator->() const { return  Cast<T*>(the_p); }

  // Is count one?
  bool unique() const { return ((is_null() == false) && the_p->unique()); }

  void increment() { if (is_null()) return; the_p->increment(); }
  void decrement() {
    if (is_null()) return;
    if (unique()) delete the_p;
    else the_p->decrement();
  }

  // Is the_p pointing to NULL?
  bool is_null() const { return the_p == 0; }

  friend
    bool operator==(const refcountable_ptr<T>& lhs, const refcountable_ptr<T>& rhs) {
    return lhs.the_p == rhs.the_p;
  }
  friend
    bool operator!=(const refcountable_ptr<T>& lhs, const refcountable_ptr<T>& rhs) {
    return lhs.the_p != rhs.the_p;
  }

 protected:
  Countable*			the_p;

 private:
  bool 				smart_behavior_;
};

#endif
