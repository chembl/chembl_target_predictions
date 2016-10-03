# -*- coding: utf-8 -*-

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import pandas as pd

import numpy
from rdkit import DataStructs

import pickle

print 'Data preparation'

data = pd.read_csv('../chembl_22/chembl_1uM.csv')

#############
###data = data.head(1000)

print "data", data.shape

mols = data[['MOLREGNO','SMILES']]
mols = mols.drop_duplicates('MOLREGNO')
mols = mols.set_index('MOLREGNO')
mols = mols.sort_index()

print "mols", mols.shape

targets = data[['MOLREGNO','TARGET_CHEMBL_ID']]
targets = targets.sort_index(by='MOLREGNO')

targets = targets.groupby('MOLREGNO').apply(lambda x: ','.join(x.TARGET_CHEMBL_ID))
targets = targets.apply(lambda x: x.split(','))
targets = pd.DataFrame(targets, columns=['targets'])

print "targets", targets.shape

PandasTools.AddMoleculeColumnToFrame(mols, smilesCol = 'SMILES')

dataset = pd.merge(mols, targets, left_index=True, right_index=True)

dataset = dataset.ix[dataset['ROMol'].notnull()]

print dataset.shape


# Learning
print 'Learning'

class FP:
  def __init__(self, fp):
        self.fp = fp
  def __str__(self):
      return self.fp.__str__()
    
def computeFP(x):
    #compute depth-2 morgan fingerprint hashed to 2048 bits
    fp = Chem.GetMorganFingerprintAsBitVect(x,2,nBits=2048)
    res = numpy.zeros(len(fp),numpy.int32)
    #convert the fingerprint to a numpy array and wrap it into the dummy container
    DataStructs.ConvertToNumpyArray(fp,res)    
    return FP(res)
        
dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)

#filter potentially failed fingerprint computations
dataset = dataset.ix[dataset['FP'].notnull()]

print 'fps done'

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

from sklearn.preprocessing import MultiLabelBinarizer

X = [f.fp for f in dataset['FP']]
yy = [c for c in dataset['targets']]

##print dataset['targets'].head()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(yy) ## this is for newer versions of sklearn

#joblib.dump(mlb, '../chembl_22/mNB_1uM_targets.pkl')

#print mlb.classes_ 

#print y

#morgan_bnb = OneVsRestClassifier(BernoulliNB())
#morgan_bnb = OneVsRestClassifier(BernoulliNB(), n_jobs=4)
###morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=8)
morgan_bnb = OneVsRestClassifier(MultinomialNB())

print 'model building'
morgan_bnb.fit(X,y)

morgan_bnb.targets = mlb.classes_

print morgan_bnb.multilabel_
print morgan_bnb.targets

#print morgan_bnb.classes_

#print morgan_bnb.label_binarizer_.classes_

joblib.dump(morgan_bnb, '../chembl_22/models/1uM/mNB_1uM_all.pkl')

##predicted = morgan_bnb.predict(X)
##print predicted
##all_labels = mlb.inverse_transform(predicted)
##print all_labels

print 'done!'

