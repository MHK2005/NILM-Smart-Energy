import pandas as pd, numpy as np, tarfile
USE = ['dataid','localminute','grid','solar','solar2','air1','dishwasher1','clotheswasher1','drye1','furnace1','oven1','kitchenapp1','refrigerator1','car1']
tf = tarfile.open('/mnt/user-data/uploads/1minute_data_austin_tar.gz','r:gz')
fh = tf.extractfile('1minute_data_austin/1minute_data_austin.csv')
parts = []
for ch in pd.read_csv(fh, usecols=USE, chunksize=500_000):
    ch['ts'] = pd.to_datetime(ch['localminute'], utc=True)
    ch = ch.drop(columns='localminute')
    ch['dataid'] = ch['dataid'].astype('int32')
    for c in ch.columns:
        if c not in ('dataid','ts'): ch[c] = ch[c].astype('float32')
    parts.append(ch)
df = pd.concat(parts, ignore_index=True); del parts
df = df.sort_values(['dataid','ts']).reset_index(drop=True)
df.to_pickle('austin_slim2.pkl')
print(df.shape, df.dataid.nunique(), 'dups', int(df.duplicated(['dataid','ts']).sum()), flush=True)
