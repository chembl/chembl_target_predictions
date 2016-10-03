# -*- coding: utf-8 -*-

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import pandas as pd
from collections import OrderedDict

import numpy
from rdkit import DataStructs

from sklearn.externals import joblib
from sklearn.metrics import classification_report


morgan_bnb = joblib.load('../chembl_22/models/1uM/mNB_1uM_all.pkl')
#mlb = joblib.load('../chembl_22/mNB_1uM_targets.pkl')

def topNpreds(m,fp,N=5):
    probas = list(morgan_bnb.predict_proba(fp)[0])
    d = dict(zip(classes,probas))
    scores = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))
    return [(m, t, s) for t, s in scores.items()[0:N]]

print morgan_bnb.multilabel_

#classes = list(morgan_bnb.classes_)
classes = list(morgan_bnb.targets)

print "targets", len(classes)

print "reading drugs..."

mols = pd.read_csv('../chembl_22/chembl_drugs.csv')

####
##mols = mols.head(10)

print mols.head()

PandasTools.AddMoleculeColumnToFrame(mols, smilesCol = 'CANONICAL_SMILES')

mols = mols.ix[mols['ROMol'].notnull()]

print mols.shape

class FP:
  def __init__(self, fp):
        self.fp = fp
  def __str__(self):
      return self.fp.__str__()

def computeFP(x):
    #compute depth-2 morgan fingerprint hashed to 1024 bits
    fp = Chem.GetMorganFingerprintAsBitVect(x,2,nBits=2048)
    res = numpy.zeros(len(fp),numpy.int32)
    #convert the fingerprint to a numpy array and wrap it into the dummy container
    DataStructs.ConvertToNumpyArray(fp,res)    
    return FP(res.reshape(1,-1))

mols['FP'] = mols.apply(lambda row: computeFP(row['ROMol']), axis=1)

#filter potentially failed fingerprint computations
mols = mols.ix[mols['FP'].notnull()]

fps = [f.fp for f in mols['FP']]

molregnos = mols['PARENT_MOLREGNO']

print "Predicting..."

ll = []

for m,f in zip(molregnos,fps):
    ll.extend(topNpreds(m,f,50))

preds = pd.DataFrame(ll,columns=['molregno','target_chembl_id','proba'])

print preds.head(10)

preds.to_csv('../chembl_22/drug_predictions_1uM.csv')

print "Done!"

