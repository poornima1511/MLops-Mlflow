import mlflow
from mlflow import MlflowClient
import duckdb 
import imaplib
import email
import re
import email
import re
import smtplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sys


email_from = "p@gmail.com"
email_list=["xyz@gmail.com.com"]
pswd = "xyz"

# Setup port number and server name
smtp_port = 587                 # Standard secure SMTP port
smtp_server = "smtp.gmail.com"  # Google SMTP Server
def mail1(file1):
    subject = f'New email from {email_from} with attachments!'
    for person in email_list:
        # Make the body of the email
        body = f"""
        Hi {person},

        Please click on one of the buttons below to respond:
        <br><br>
        <a href="http://localhost:8080/job/pipeline_test/">Jenkins Dashboard</a>
        """

        # Create the MIME message
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = person
        msg['Subject'] = subject

        # Attach the HTML body
        msg.attach(MIMEText(body, 'html'))
        # filename = "result.txt"
        # print(filename)
        # print(file1)
        # Open the file in python as a binary
        attachment= open(file1, 'rb')  # r for read and b for binary

        # Encode as base 64
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + "result.txt")
        msg.attach(attachment_package)
        # text = msg.as_string()


        # Connect with the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(email_from, pswd)
        print("Successfully connected to server")
        print()

        # Send emails to "person" as the list is iterated
        print(f"Sending email to: {person}...")
        TIE_server.sendmail(email_from, person, msg.as_string())
# name the email subject
def mail3(file1):
    subject = f'New email from {email_from} with attachments!'
    for person in email_list:
        # Make the body of the email  
        #location of jenkins pipeline where the user will click to confirm
        body = f"""
        Hi {person},

        Please click on one of the buttons below to respond:
        <br><br>
        <a href="http://localhost:8080/job/pipeline_test(version2)/">Jenkins Dashboard</a>   
      
        """

        # Create the MIME message
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = person
        msg['Subject'] = subject

        # Attach the HTML body
        msg.attach(MIMEText(body, 'html'))
        # filename = "result.txt"
        # print(filename)
        # print(file1)
        # Open the file in python as a binary
        attachment= open(file1, 'rb')  # r for read and b for binary

        # Encode as base 64
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + "result.txt")
        msg.attach(attachment_package)
        # text = msg.as_string()


        # Connect with the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(email_from, pswd)
        print("Successfully connected to server")
        print()

        # Send emails to "person" as the list is iterated
        print(f"Sending email to: {person}...")
        TIE_server.sendmail(email_from, person, msg.as_string())
# project_name=input("project_name:")
project_name=sys.argv[1]
mlflow.set_tracking_uri("http://localhost:5000")
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments(filter_string=f"name = '{project_name}'")]
# Search for the deployed model run


if project_name=='regression':

    runs = mlflow.search_runs(experiment_ids=all_experiments, order_by=[ 'metrics.training_r2_score DESC'])  # we need this with respect to metrix
    print(runs.columns)
    # Extract the deployed model version
    model_version = runs[['run_id','metrics.training_r2_score']]
    # print(type(model_version))       
    # print(model_version)     

    prod_runid = []
    client = MlflowClient()  # it will intialize the mlflow client connect with mlflow (is there is some change then we need to connect)
    for mv in client.search_model_versions(f"name='{project_name}'"):
        d = dict(mv)
        if d['current_stage'] == 'Production':                      
            prod_runid.append(d['run_id'])      #getting run id of current prodcution version
    if len(prod_runid) == 0 :
        best_model_runid = model_version.loc[0]['run_id']   
        client.create_registered_model(f"{project_name}")   # if there is not model in production then we need to create a modle version with name and the runid of that model
        result = client.create_model_version(
                name=f"{project_name}",
                source=f'mlflow-artifacts:/0/{best_model_runid}/artifacts/model',
                run_id=f'{best_model_runid}',
            )
        
        mail1("C:/ProgramData/Jenkins/.jenkins/workspace/pipeline_test(version2)/result.txt")
        client.transition_model_version_stage(  
                name=f"{project_name}", version=1, stage="Production")    # after that pushing the model into production  because it is the first model        
    # print(prod_runid)
    # test_d = prod_runid[0]
    else : 
        get={'run_id' : [], 'version':[],'current_stage':[]}
        import pandas as pd
        # neww=pd.DataFrame(columns=['run_id','version'])
        for mv in client.search_model_versions(f"name='{project_name}'",order_by=['version_number']):
            get_runid=dict(mv)
            get['run_id'].append(get_runid['run_id'])
            get['version'].append(get_runid['version'])
            get['current_stage'].append(get_runid['current_stage'])

        get1=pd.DataFrame.from_dict(get)
        # print("------")
        if prod_runid[0] != model_version.loc[0]['run_id']:
            # print('if is executed..')
            best_model_rmse = model_version.loc[0]['metrics.training_r2_score']
            best_model_runid = model_version.loc[0]['run_id']
            conn =  duckdb.connect()
            df = model_version
            prod_model_rmse = conn.execute(f"""
            select 'metrics.training_r2_score' from df where "run_id" == '{prod_runid[0]}' ;
            """).df()
            prod_model_rmse=prod_model_rmse.loc[0]['metrics.training_r2_score']
            print("production : ",prod_model_rmse)
            print('new Model : ',best_model_rmse)

            r2_change = best_model_rmse - prod_model_rmse
            r2_percentage_change = (r2_change / prod_model_rmse)*100
            
            print('rmse_change : ',r2_change)
            print('rmse_percentage_change : ',abs(r2_percentage_change))
        

            if abs(r2_percentage_change) > 10 :   # threshold fixed need to updated
                result = client.create_model_version(
                    name=f"{project_name}",
                    source=f'mlflow-artifacts:/0/{best_model_runid}/artifacts/model',
                    run_id=f'{best_model_runid}',
                )
                res_file = [f"'production r2' : {prod_model_rmse} \n",f"'new Model' : {best_model_rmse} \n",
                        f"'r2_change' : {r2_change} \n",f"'r2_percentage_change' : {abs(r2_percentage_change)} \n"]
                file1 = open('result.txt', 'w')
                file1.writelines(res_file)
                file1.close()
                if len(get1["version"])<2:
                    client.transition_model_version_stage(
                        name=f"{project_name}", version=int(get1['version'].iloc[-1]), stage="Production")
                else:
                    client.transition_model_version_stage(
                        name="f{project_name}", version=int(get1['version'].iloc[-1]), stage="Archived")
                client.transition_model_version_stage(
                    name="f{project_name}", version=int(get1['version'].iloc[-1])+1, stage="Staging")
            
                mail3("C:/ProgramData/Jenkins/.jenkins/workspace/pipeline_test(version2)/result.txt")  
            else:
                print("Production model is already performing good..")
                raise Exception("Production model is already performing good..")
        else :
            print("Production model is already performing good..")
            raise Exception("Production model is already performing good..")
               
elif project_name=="classification":
    runs = mlflow.search_runs(experiment_ids=all_experiments, order_by=['metrics.training_accuracy_score DESC'])  
    print(runs.columns)
    # Extract the deployed model version
    model_version = runs[['run_id','metrics.training_accuracy_score']]
    # print(type(model_version))       
    # print(model_version)     
    prod_runid = []
    client = MlflowClient()  # it will intialize the mlflow client connect with mlflow (is there is some change then we need to connect)
    for mv in client.search_model_versions(f"name='{project_name}'"):
        d = dict(mv)
        if d['current_stage'] == 'Production':
            prod_runid.append(d['run_id'])
    if len(prod_runid) == 0 :
        best_model_runid = model_version.loc[0]['run_id']
        client.create_registered_model(f'{project_name}')
        result = client.create_model_version(
                name=f'{project_name}',
                source=f'mlflow-artifacts:/0/{best_model_runid}/artifacts/model',
                run_id=f'{best_model_runid}',
            )
        
        mail1("C:/ProgramData/Jenkins/.jenkins/workspace/pipeline_test(version2)/result.txt")
        client.transition_model_version_stage(
                name=f'{project_name}', version=1, stage="Production")
        
    # print(prod_runid)
    # test_d = prod_runid[0]
    else :
        get={'run_id' : [], 'version':[],'current_stage':[]}
        import pandas as pd
        # neww=pd.DataFrame(columns=['run_id','version'])
        for mv in client.search_model_versions(f"name='{project_name}'",order_by=['version_number']):
            get_runid=dict(mv)
            get['run_id'].append(get_runid['run_id'])
            get['version'].append(get_runid['version'])
            get['current_stage'].append(get_runid['current_stage'])

        get1=pd.DataFrame.from_dict(get)
        # print("------")
        if prod_runid[0] != model_version.loc[0]['run_id']:
            # print('if is executed..')
            best_model_rmse = model_version.loc[0]['metrics.training_accuracy_score']
            best_model_runid = model_version.loc[0]['run_id']
            conn =  duckdb.connect()
            df = model_version
            prod_model_rmse = conn.execute(f"""
            select "metrics.training_accuracy_score" from df where "run_id" == '{prod_runid[0]}' ;
            """).df()
            prod_model_rmse=prod_model_rmse.loc[0]['metrics.training_accuracy_score']
            print("production : ",prod_model_rmse)
            print('new Model : ',best_model_rmse)

            rmse_change = best_model_rmse - prod_model_rmse
            rmse_percentage_change = (rmse_change / prod_model_rmse)*100
            
            print('rmse_change : ',rmse_change)
            print('rmse_percentage_change : ',abs(rmse_percentage_change))
        

            if abs(rmse_percentage_change) > 10 :   # threshold fixed need to updated
                result = client.create_model_version(
                    name=f"{project_name}",
                    source=f'mlflow-artifacts:/0/{best_model_runid}/artifacts/model',
                    run_id=f'{best_model_runid}',
                )
                res_file = [f"'production accuracy' : {prod_model_rmse} \n",f"'new Model' : {best_model_rmse} \n",
                        f"'accuracy_change' : {rmse_change} \n",f"'rmse_percentage_change' : {abs(rmse_percentage_change)} \n"]
                file1 = open('result.txt', 'w')
                file1.writelines(res_file)
                file1.close()
                if len(get1["version"])<2:
                    client.transition_model_version_stage(
                        name=f"{project_name}", version=int(get1['version'].iloc[-1]), stage="Production")
                else:
                    client.transition_model_version_stage(
                        name=f"{project_name}", version=int(get1['version'].iloc[-1]), stage="Archived")
                client.transition_model_version_stage(
                    name=f"{project_name}", version=int(get1['version'].iloc[-1])+1, stage="Staging")
            
                mail3("C:/ProgramData/Jenkins/.jenkins/workspace/pipeline_test(version2)/result.txt")
            else:
                print("Production model is already performing good..")
                raise Exception("Production model is already performing good..")
        else :
            print("Production model is already performing good..")
            raise Exception("Production model is already performing good..")
            