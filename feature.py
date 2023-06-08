from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.datasets import load_breast_cancer
import sys
import pandas as pd
project_name=sys.argv[1]
def feature(data_X, Y):
    print("feature exectuting")                    
    if project_name=="regression":               
        method=RandomForestRegressor()     
        sfs = SFS(RandomForestRegressor(),                 
                k_features=(1,data_X.shape[1]),
                forward=True,
                floating=False,
                scoring='neg_mean_squared_error',
                cv=5)
        sfs.fit(data_X, Y)
        performance = sfs.get_metric_dict()
        num_features = []
        scores = []
        for k, v in performance.items():
            num_features.append(k)
            scores.append(-v['avg_score'])  
        optimal_num_features = num_features[scores.index(min(scores))]
        print('Optimal Number of Features:', optimal_num_features)
    else:
        method=RandomForestClassifier()
        data_X=pd.DataFrame(data_X)
        sfs = SFS(RandomForestClassifier(),
                                k_features=(1, len(data_X.columns)),
                                forward=True,
                                floating=False,
                                scoring='accuracy',
                                cv=5)
        sfs.fit(data_X,Y)
        performance = sfs.get_metric_dict()
        num_features = []
        scores = []
        for k, v in performance.items():
            num_features.append(k)
            scores.append(v['avg_score'])  
        optimal_num_features = num_features[scores.index(max(scores))]
        print('Optimal Number of Features:', optimal_num_features)

    sfs_for= SFS(method,
         k_features=optimal_num_features,                    
         forward=True,                               
         floating=False,
         cv=0)
    sfs_for.fit(data_X, Y)

    sfs_back= SFS(method,
         k_features=optimal_num_features,                    
         forward=True,                               
         floating=True,
         cv=0)
    sfs_back.fit(data_X, Y)


    sfs_exa= SFS(method,
         k_features=optimal_num_features,                    
         forward=False,                               
         floating=False,
         cv=0)
    sfs_exa.fit(data_X, Y)
    new_features=list(set(sfs_for.k_feature_names_) | set(sfs_back.k_feature_names_) |set(sfs_exa.k_feature_names_))
    print("end")
    return(new_features)


if project_name=="regression":
        df=pd.read_csv("sale.csv")
        df["deal_id"]=df["deal_id"].apply(hash) 
        df["start_date"]=df["start_date"].apply(hash)
        X_regression=df.drop("treatment",axis=1)
        y_regression = df["treatment"]
        new_feature=feature(X_regression,y_regression)
        X_regression=X_regression[new_feature]
        X_regression=pd.DataFrame(X_regression)
        X_regression.to_csv('regree.csv',index=False)
else:
    data = load_breast_cancer()  
    X_classification = data.data
    y_classification = data.target
    new_feature=feature(X_classification,y_classification)
    X_classification=X_classification[:,new_feature]
    X_classification=pd.DataFrame(X_classification)
    X_classification.to_csv('classifi.csv')