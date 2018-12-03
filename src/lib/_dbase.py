#!/usr/bin/python

import anydbm
import cPickle

class Dbase:
  def __init__(self, spkFile, uttFile, spkMode="r", uttMode="r",
         spkKey="EPISODE", uttKey="SEGS"):
    self.spk = SpkDbase(spkFile, spkMode, uttKey)
    self.utt = UttDbase(uttFile, uttMode, spkKey)
    self.spkKey = spkKey
    self.uttKey = uttKey

  def uttInfo(self, spkID, uttID):
    """
    get information of given speaker and utterance
    """
    return {spkID: self.spk[spkID], uttID: self.utt[uttID]}


class GenericDbase:
  def __init__(self, datafile, mode="r"):
    """
    open database:
    datafile:  name of database file
    indexfile: name of index file
    mode : r | rw | rwc
    """
    try:
      self.db = anydbm.open(datafile, mode)
    except ImportError:
      import dbm
      self.db = dbm.open(datafile, mode)
      
  def __getitem__(self, key):
    return cPickle.loads(self.db[key]) 

  def __setitem__(self, key, data):
    self.db[key] = cPickle.dumps(data)

  def close(self):
    "close database"
    self.db.close();

  def add(self, key, data):
    """
    add record to database
    data: any python data to be stored in the dbase
    """
    self[key] = data  

  def delete(self, key):
    "delete record from database"
    del self.db[key]

  def get(self, key):
    "get record from database"
    return self.db[key]

  def first():
    "get first key in database"
    pass

  def next():
    "get next key in database"
    pass

  def list(self):
    "list all keys in database"
    return self.db.keys()


class SpkDbase(GenericDbase):
  def __init__(self, datafile, mode="r", uttKey="SEGS"):
    GenericDbase.__init__(self, datafile, mode)
    self.uttKey = uttKey

  def foreachSegment(self, speaker, body):
    """
    foreachSegment is a for loop over all segments of a a speaker.
    The segment list is obtained from the database.
    speaker:   key in the DB for the speaker
    body:    callback that is calles for each segment
    """
    spkInfo = self.db[speaker][self.uttKey]
    for utt in spkInfo:
      body(utt)


class UttDbase(GenericDbase):
  def __init__(self, datafile, mode="r", spkKey="EPISODE"):
    GenericDbase.__init__(self, datafile, mode)
    self.spkKey = spkKey

  def getSpeaker(uttID):
    """
    find speaker of given utterance
    """
    return self.db[uttID][self.spkKey]















