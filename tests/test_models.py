import unittest
import pandas as pd
import numpy as np
import stanscofi.models

def fold2mask(fold):
    return pd.DataFrame(fold, columns=list('ijv')).pivot(
        index='i', columns='j')

class TestFolds(unittest.TestCase):

    def test_get_folds(self):
        df = pd.DataFrame(np.random.randint(2, size=(6, 5)))
        print(df)
        # folds = get_folds(df)
        folds = get_folds_disjoint_diseases(df)
        train, test = folds[0]
        print(train)
        print('train', fold2mask(train))
        print('test', fold2mask(test))
        print(len(train), len(test))
        self.assertEqual('hello', 'hello')
