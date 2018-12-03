# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.dbase
#   Purpose: Database managment.
#   Author:  Fabian Jakobs

#/usr/bin/env python

from __future__ import generators
import os
import string
import _dbase

import anydbm
import cPickle
import string
import re

class Dbase:
    """
    Database class to handle the speaker and utterance databases

    Since Janus Databases are not uniform it might me neccecary to
    overwrite the set-/getSpeaker and set-/getUtternce methods for
    a special database.
    """
    
    def __init__(self, spkFile, uttFile, spkMode="r", uttMode="r"):
        """
        load Database from spkFile and uttFile
        """
        self._db = _dbase.Dbase(spkFile, uttFile, spkMode, uttMode)

    def getSpeaker(self, SID):
        """
        returns a speaker object with the specified speaker ID (SID) 
        """
        db = self._db.spk[SID]
        uids = db["SEGS"]
        return Speaker(SID, self, uids, db)

    def setSpeaker(self, SID, spk):
        """
        writes a speakerobject back into the database
        """
        db_entry = spk.data
        db_entry["SPK"] = (spk.sid,)
        db_entry["SEGS"] = spk.utt.spkKeys
        self.spk[self.sid] = db_entry

    def spkKeys(self):
        """
        lists all speaker keys
        """
        return self._db.spk.list()

    def spkIterator(self):
        """
        iterator over all speakers in the database

        Example:
        >>> for speaker in database.spkIterator():
        >>>     print speaker
        """
        for key in self.spkKeys():
            yield self.getSpeaker(key)

    def getUtterance(self, UID):
        """
        returns an utterance object with the specified utterance ID (SID) 
        """
        db = self._db.utt[UID]
        return Utterance(UID,
                         self,
                         db["DIALOGUE"][0]+ "_" + db["SPEAKER"][0],
                         os.path.join(db["PATH"][0], db["FILE"][0]),
                         int(db["FROM"][0]),
                         int(db["TO"][0]),
                         db)

    def setUtterance(self, UID, utt):
        """
        wirtes an utterance back into the database
        """
        db_entry = utt.data
        db_entry["UTT"] = utt.uid
        (dialogue, speaker) = String.split(utt.sid, "_", 1)
        db_entry["DIALOGUE"] = dialogue
        db_entry["SPEAKER"] = speaker
        (path, file) = os.path.split(utt.file)
        db_entry["PATH"] = path
        db_entry["FILE"] = file        
        db_entry["FROM"] = self.cfrom
        db_entry["TO"] = self.cto

    def uttKeys(self):
        """
        returns all keys of utterances in the database
        """
        return self._db.utt.list()

    def uttIterator(self):
        """
        iterator over all utterances in the database

        example:
        >>> for utterance in database.uttIterator():
        >>>    print utterance
        """
        for key in self._db.utt.list():
            yield self.getUtterance(key)

class DB200x(Dbase):
    """
    DB200x Database

    this is an example how to customize the database module for a special
    database. The dafault databse doesn't know about the right field names
    of the utterance database so we have to overwrite the methods converting
    database objects to Python objects and vice versa.
    """

    def getUtterance(self, UID):
        """
        returns an utterance object with the specified utterance ID (SID) 
        """
        db = self._db.utt[UID]
        return Utterance(UID,
                         self,
                         db["SPEAKER"][0],
                         os.path.join(db["PATH"][0], db["ADC"][0]),
                         0, -1, db)

    def setUtterance(self, uid, utt):
        raise NotImplementedError, ""


class UttList:
    """
    Uttlist stores a list of all utterances belonging to a speaker.
    It behaves like a normal Python list but reads the data from
    the database.
    """
    def __init__(self, dbase, UIDS):
        self._db = dbase
        self._utt = UIDS

    def __getitem__(self, key):
        return self._db.getUtterance(key)

    def __setitem__(self, key, value):
        self._db.setUtterance(key, value)

    def __len__(self):
        return len(self._utt)

    def __iter__(self):
        for utt in self._utt:
            yield self._db.getUtterance(utt)

    def keys(self):
        return self._utt

class Speaker:
    def __init__(self, SID, dbase, UIDS=None, data=None):
        self.sid = SID
        self.utt = UttList(dbase, UIDS)
        self.data = data

    def __str__(self):
        return """Speaker:\t%s
        UIDs: \t%s""" % (self.sid, self.utt.keys())

class Utterance:
    def __init__(self, UID, dbase, SID, file="", cfrom=0, cto=-1, data=None):
        self.uid = UID
        self.dbase = dbase
        self.sid = SID
        self.file = file
        self.cfrom = cfrom
        self.cto = cto
        self.data = data

    def __str__(self):
        return """ Utterance:\t%s
        Speaker:\t%s
        File:\t%s""" % (self.uid, self.sid, self.file)

    def getSample(self, prefix=""):
        """
        returns a feature Object. prefix and self.file build the
        filename of the file containing the feature.
        """
        import sound
        import os.path
        return sound.SoundBuffer(os.path.join(prefix, self.file),
                          cfrom=self.cfrom,
                          cto=self.cto)

def split_braces(line):
    open_br = []
    close_br = []
    i = 0
    for c in line:
        if c == '{':
            open_br.append(i)
        elif c == '}':
            close_br.append(i)
        i += 1        
    retval = []
    if len(open_br) != len(close_br):
        raise RuntimeError("unequal amount of open and closing brackets.")
    for i in range(len(open_br)):
        retval.append(line[open_br[i]+1: close_br[i]])
    return retval

def importJanusDB(filename, key="SPK"):
    file = open(filename)
    db = {}
    for line in file:
        db_entry = {}
        line = string.strip(line)
        for entry in split_braces(line):
            list = string.split(entry)
            db_entry[list[0]] = list[1:]
        db[db_entry[key][0]] = db_entry 
    return db

def convert(infile, outfile, key="SPK"):
    db = anydbm.open(outfile, "n")
    jdb = importJanusDB(infile, key)
    for key in jdb.keys():
        db[key] = cPickle.dumps(jdb[key], 1)
