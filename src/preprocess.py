import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_data():
    train_df = pd.read_csv('data/mitbih_train.csv' , header=None)
    test_df = pd.read_csv('data/mitbih_test.csv', header=None)

    train_df.columns = [f'feature_{i}' for i in range(187)] + ['label']
    test_df.columns = [f'feature_{i}' for i in range(187)] + ['label']

    return train_df, test_df

def binaryConverter(train_df, test_df):
    train_df['label'] = train_df['label'].apply(lambda x:0 if x==0.0 else 1)
    test_df['label'] = test_df['label'].apply(lambda x:0 if x==0.0 else 1)

    return train_df, test_df

def balancing_data(train_df):
    #Upsampling the abnormal to normal
    train_df_normal = train_df[train_df['label']==0]
    train_df_abnormal = train_df[train_df['label']==1]

    upsampled = resample(
        train_df_abnormal, replace=True,
        n_samples=len(train_df_normal), random_state=42
    )
    train_df_sampled = pd.concat([train_df_normal, upsampled])
    return train_df_sampled

def Scaling_data(train_df_sampled, test_df):
    X_train = train_df_sampled.drop('label', axis=1)
    y_train= train_df_sampled['label']

    X_test = test_df.drop('label', axis=1)
    y_test= test_df['label']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
