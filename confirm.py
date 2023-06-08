import mlflow
from mlflow import MlflowClient
import duckdb

conn=duckdb.connect()
mlflow.set_tracking_uri("http://localhost:5000")

get={'run_id' : [], 'version':[],'current_stage':[]}

# neww=pd.DataFrame(columns=['run_id','version'])

client = MlflowClient() 
for mv in client.search_model_versions("name='sales_prediction_model'",order_by=['version_number']):
    get_runid=dict(mv)
    get['run_id'].append(get_runid['run_id'])
    get['version'].append(get_runid['version'])
    get['current_stage'].append(get_runid['current_stage'])
# print(get)
import pandas as pd
df=pd.DataFrame.from_dict(get)

print(df)
prod_run=conn.execute(''' 
select run_id,version from df where "current_stage"='Production';
''').df()
stage_run=conn.execute(''' 
        select run_id,version from df where "current_stage"='Staging';
        ''').df()
#print(prod_run)
# print(type(prod_run["version"][0])) 
stg=[]
for x in df["current_stage"]:
      if x=="Staging":
            stg.append(x) 
if len(stg)==1:
    # if prod_run["version"][0]!='1' :
        client.transition_model_version_stage(
        name="sales_prediction_model", version=prod_run["version"][0], stage="Archived"
        )
        client.transition_model_version_stage(
        name="sales_prediction_model", version=stage_run["version"][0], stage="Production"
        )
        
        
        # #print(prod_run)
        # print(stage_run["version"][0])
        
