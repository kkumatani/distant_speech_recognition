/**
 * @file mlist.i
 * @brief List operation
 * @author Fabian Jakobs
 */

#ifndef MLIST_H
#define MLIST_H

#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>

#include "common/common.h"
#include "common/jexception.h"

/*
#include "common/refcount.h"
*/

using namespace std;

#define Warning if(setErrLine_(__LINE__, __FILE__)) warnMsg_
extern int setErrLine_(int line, const char* file);
extern void warnMsg_(const char* message, ...);

// ----- definition of class `String' -----
//
class String : public string {
 public:
  String () : string("") {}
  String (const char* s) : string(((s == NULL) ? "" : s)) {}
  String (const string& s) : string(s) {}

  operator const char*() const { return c_str(); }
  const char* chars() const { return c_str(); }
};


char* date_string(void);
void split_list(const String& line, std::list<String>& out);

typedef unsigned short UnShrt;
typedef float  LogFloat;
typedef double LogDouble;


// ----- definition of class template `List' -----
//
template <class Type, class Key = String>
class List {
  typedef vector<Type>   	 		ListVector_;
  typedef typename ListVector_::iterator	ListVectorIterator_;
  typedef typename ListVector_::const_iterator	ListVectorConstIterator_;
  typedef map< Key, Type> 	 		TypeMap_;
  typedef typename TypeMap_::iterator	 	TypeMapIterator_;
  typedef typename TypeMap_::const_iterator	TypeMapConstIterator_;
  typedef map< Key, unsigned>	 		IndexMap_;
  typedef typename IndexMap_::iterator	 	IndexMapIterator_;
  typedef typename IndexMap_::const_iterator	IndexMapConstIterator_;

 public:
  List(const String& nm) : name_(nm) { }

  const String& name() const { return name_; }

  inline unsigned add(const Key& key, Type item);

  Type& operator[](unsigned index) {
    assert(index < listVector_.size());
    return listVector_[index];
  }
  const Type& operator[](unsigned index) const {
    assert(index < listVector_.size());
    return listVector_[index];
  }
  Type& operator[](const Key& key) {
    TypeMapIterator_ itr = typeMap_.find(key);
    if (itr == typeMap_.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), name_.c_str());
    return (*itr).second;
  }
  const Type& operator[](const Key& key) const {
    TypeMapConstIterator_ itr = typeMap_.find(key);
    if (itr == typeMap_.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), name_.c_str());
    return (*itr).second;
  }
  unsigned index(const Key& key) {
    IndexMapIterator_ itr = indexMap_.find(key);
    if (itr == indexMap_.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), name_.c_str());
    return (*itr).second;
  }
  unsigned index(const Key& key) const {
    IndexMapConstIterator_ itr = indexMap_.find(key);
    if (itr == indexMap_.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), name_.c_str());
    return (*itr).second;
  }
  bool isPresent(const Key& key) const {
    IndexMapConstIterator_ itr = indexMap_.find(key);
    return itr != indexMap_.end();
  }

  unsigned size() const { return listVector_.size(); }
  void clear() { listVector_.clear();  typeMap_.clear();  indexMap_.clear(); }

  class Iterator;  	friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

 private:
  unsigned firstComp_(int nParts, int part) const;
  unsigned lastComp_(int nParts, int part) const;

  const String					name_;

  ListVector_					listVector_;
  TypeMap_					typeMap_;
  IndexMap_					indexMap_;
};

template <class Type, class Key>
unsigned List<Type,Key>::add(const Key& key, Type item)
{
  unsigned idx = size();

  indexMap_[key] = idx;
  typeMap_[key]  = item;
  listVector_.push_back(item);

  return idx;
}

template <class Type, class Key>
unsigned List<Type,Key>::firstComp_(int nParts, int part) const
{
  unsigned segment = listVector_.size() / nParts;

  return (part - 1) * segment;
}

template <class Type, class Key>
unsigned List<Type,Key>::lastComp_(int nParts, int part) const
{
  unsigned ttlComps = listVector_.size();
  unsigned segment  = ttlComps / nParts;

  return (part == nParts) ? (ttlComps - 1) : (part * segment - 1);
}

template <class Type, class Key>
class List<Type,Key>::Iterator {
 public:
  Iterator(List& lst, int nParts = 1, int part = 1) :
    beg_(lst.listVector_.begin() + lst.firstComp_(nParts, part)),
    cur_(lst.listVector_.begin() + lst.firstComp_(nParts, part)),
    end_(lst.listVector_.begin() + lst.lastComp_(nParts, part) + 1) { }

  bool more() const { return cur_ != end_; }
  void operator++(int) {
    if (more()) cur_++;
  }
  // Type& operator->() { return *cur_; }
        Type& operator*()       { return *cur_; }
  const Type& operator*() const { return *cur_; }

 private:
  ListVectorIterator_				beg_;
  ListVectorIterator_				cur_;
  ListVectorIterator_				end_;
};

template <class Type, class Key>
class List<Type,Key>::ConstIterator {
 public:
  ConstIterator(const List& lst, int nParts = 1, int part = 1) :
    beg_(lst.listVector_.begin() + lst.firstComp_(nParts, part)),
    cur_(lst.listVector_.begin() + lst.firstComp_(nParts, part)),
    end_(lst.listVector_.begin() + lst.lastComp_(nParts, part) + 1) { }

  bool more() const { return cur_ != end_; }
  void operator++(int) {
    if (more()) cur_++;
  }
  const Type& operator*() const { return *cur_; }

 private:
  ListVectorConstIterator_			beg_;
  ListVectorConstIterator_			cur_;
  ListVectorConstIterator_			end_;
};

/* adds alls items of the file and passes them as string to T's addmethod->__add */
template<class T>
void freadAdd(const String& fileName, char commentChar, T* addmethod)
{
  FILE* fp = btk_fopen(fileName,"r");

  if (fp == NULL)
    throw jio_error("Can't open file '%s' for reading.\n", fileName.c_str());

  cout << "Reading: " << fileName << endl;

  static char line[100000];
  while (1) {
    list<string> items;
    items.clear();
    char* p;
    int   f = fscanf(fp,"%[^\n]",&(line[0]));

    assert( f < 100000);

    if      ( f <  0)   break;
    else if ( f == 0) { fscanf(fp,"%*c"); continue; }

    if ( line[0] == commentChar) continue;

    for (p=&(line[0]); *p!='\0'; p++)
      if (*p>' ') break; if (*p=='\0') continue;

    try {
      // cout << "Adding: " << line << endl;
      addmethod->__add(line);
    } catch (j_error) {
      // cout << "Closing file ...";
      btk_fclose( fileName, fp);
      // cout << "Done" << endl;
      throw;
    }
  }
  btk_fclose( fileName, fp);
}

#endif
