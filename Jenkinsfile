pipeline{
    agent any
  
  stages {  
  stage('Feature selection'){
      steps{
                bat 'pip install scikit-learn'
                bat 'pip install pandas'
                bat 'pip install mlflow'
                bat 'pip install duckdb'
                bat 'pip install mlxtend'
                bat "python feature.py ${Model_type}"
                
      }
  }
  stage('Training the model'){
      
            steps{
               bat "python model.py ${Model_type}"
                }
    }
    stage('Choosing the best model'){
      
            steps{
                bat "python choosing_test.py ${Model_type}"
               }
           
     }
       
        stage("Confirmation for Deployment") {
        input {
                message "Ready to deploy?"
                ok "Yes"
            }
        steps{
                echo  'Ready to deploy'
                
            }
           }   
    stage("Deployment") {
            steps{
                bat 'python confirm.py'

            }
          
        }
}
}
