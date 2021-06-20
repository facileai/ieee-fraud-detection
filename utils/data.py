from fastai.basics import *
from fastai.tabular.all import *
import random

@patch
def export(self:TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to
        
def get_data(data_pth, valid_pct=0.2, hp=True):
    
    df = pd.read_csv(data_pth, low_memory=False)

   

    with open('artifacts/features.txt') as json_file:
            features = json.load(json_file)
    
    if hp: 
        cont = features['cont']
        cat = features['cat']
        dep_var = features['dep_var']
        cols = cat+cont+[dep_var]

        df = df[cols]
        df = df.sort_values(by=['TransactionDT'])
        pct = int(df.shape[0]*(100-(valid_pct*100))/100)
        splits = (list(df[:pct].index),list(df[pct:].index))
    #     splits = RandomSplitter(valid_pct=valid_pct)(range_of(df))

    else:

        cont = features['cont']
        cat = features['cat']
        dep_var = features['dep_var']
        cols = cat+cont+[dep_var]

        df = df[cols]
        df = df.sort_values(by=['TransactionDT'])
        splits = None
        
    procs_nn = [Categorify, FillMissing]
    data_proc = TabularPandas(df, procs_nn, cat, cont, splits=splits, y_names=dep_var)
    data_proc.export('artifacts/data-proc.pkl')



    X_train, y_train, X_valid, y_valid = data_proc.train.xs, data_proc.train.y, data_proc.valid.xs, data_proc.valid.y
    

    return X_train, y_train, X_valid, y_valid, data_proc

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))

def ContToOrdinal(df,col,col_map):
   
    return df[col].map(col_map)


def balanceSample(X,y):
    df = pd.concat([X,y],axis=1)
    l = [0,0,0,3600*24*7,3600*24*14,3600*24*21,3600*24*30]
    random.shuffle(l)
    rdm = random.choice(l)
    
    df = df[df['TransactionDT'] <= df['TransactionDT'].max()-rdm]
    pos = df[df['isFraud']==1]
    neg = df[df['isFraud']==0]
    neg = neg.sort_values(by=['TransactionDT'])
    count_pos = pos.shape[0]
    neg_sample = neg.sample(count_pos*3,replace=True)
    neg_sample = neg_sample.drop_duplicates(keep='last')
    sample = pd.concat([neg_sample,pos])
    sample = sample.sample(frac=1).reset_index(drop=True)
    y = sample['isFraud'].copy()
    sample.drop(['isFraud'],axis=1,inplace=True)
   
    print(f'X shape : {sample.shape}, y shape: {y.shape}')
    return sample, y