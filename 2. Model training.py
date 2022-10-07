# Databricks notebook source
# MAGIC %pip install mlflow==1.29.0 pandas-profiling==3.3.0

# COMMAND ----------

# MAGIC %md
# MAGIC # ML model training
# MAGIC The goal of this notebook is to load the German Credit Data dataset and train two machine learning models with it. 
# MAGIC 
# MAGIC The machine learning model will be able to predict the risk of providing loans to different people.
# MAGIC 
# MAGIC This is the typical flow that a Data Scientist would follow.
# MAGIC 
# MAGIC In the high level diagram, we will work in the highlighted section:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_2.png?raw=true" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Load libraries and set the MLflow experiment
# MAGIC Please make sure that the following package is installed in your cluster (you can use the Maven repo coordinates):
# MAGIC 
# MAGIC org.mlflow:mlflow-spark:1.17.0

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pyspark.ml.functions import vector_to_array
from pandas_profiling import ProfileReport
from pyspark.sql import SparkSession

spark = (SparkSession.builder
            .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.17.0")
            .master("local[*]")
            .getOrCreate())

# Use MLflow to track experiments
experiment_name = "german_credit_experiment"
current_folder = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-1])
#mlflow.set_experiment("{}/{}".format(current_folder, experiment_name))

# COMMAND ----------

# if "Repos" == current_folder.split("/")[1]:
#   current_folder.replace("Repos", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the German Credit Data dataset
# MAGIC This dataset contains 1000 rows and 11 columns. Each row represents one person requesting a credit. We are going to use only the first 600 rows for model training. The remaining 400 rows will be used for A/B testing (we will assume these rows come at a later point in time). Regarding the columns:
# MAGIC - The *risk* column shows if it is risky (bad) or if it is not risky (good) to provide the credit
# MAGIC - The *id* column represents the unique id of the request
# MAGIC - The rest of the columns are properties of the person requesting the credit, as well as information about the requested credit itself (amount, duration, ...)

# COMMAND ----------

df = spark.read.table("default.german_credit_data").where(F.col("id") < 600) # Load only the first 600 rows
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploratory Analysis
# MAGIC In this section we will explore our dataset. This could be done in many different ways. Since this dataset is rather small, we will convert a sample of the dataframe to Pandas and use the ProfileReport library to automatically generate insights about the dataset.

# COMMAND ----------

df_profile = ProfileReport(
  df.sample(withReplacement=False, seed=42, fraction=0.5).toPandas(),
  title="Profiling Report",
  progress_bar=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling, evaluation and tracking with MLflow
# MAGIC Great! Now that we have generated insights about the dataset and it is already structured we can proceed with the modelling.
# MAGIC 
# MAGIC Modelling is a very iterative process that might need a lot of experimentation. As shown in the diagram below. It requires preparing the data, extracting features, training a model with some specific hyperparameters and evaluate it. If the evaluation shows that the model does not meet the desired level of quality, a new iteration should be done, which might imply processing the data again, extract different or more features, training the same model with different hyperparameters or a different model, and carry out the evaluation once again.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://cdn-images-1.medium.com/max/1600/1*WjXHRFcFT--7jPRWJ9Q5Ww.jpeg" width="1000"/>
# MAGIC 
# MAGIC This process can be complex to track, and also can happen over different days, weeks or even months. For that reason MLflow is a handy framework to track the experiments:
# MAGIC 
# MAGIC https://www.mlflow.org/docs/latest/tracking.html
# MAGIC 
# MAGIC In this notebook we will use MLflow to track the training of this model:
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess the data
# MAGIC We will split the original dataset into training and test. Also we will vectorize the categorical features.

# COMMAND ----------

df_train, df_test = df.randomSplit(weights=[0.8, 0.2], seed=42)

string_cols = ["sex", "housing", "saving_accounts", "checking_account", "purpose"]
strings_cols_index = [i + " encoded" for i in string_cols]
strings_cols_encoded = [i + " encoded" for i in strings_cols_index]

str_indexer_label = StringIndexer(
  inputCol="risk",
  outputCol="label"
)
str_indexer = StringIndexer(
  inputCols=string_cols,
  outputCols=strings_cols_index
)
hot_encoder = OneHotEncoder(
  inputCols=strings_cols_index,
  outputCols=strings_cols_encoded
)
vector_assembler = VectorAssembler(
  inputCols=["age", "job", "duration", "credit_amount"] + strings_cols_encoded,
  outputCol="features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper function to evaluate the models

# COMMAND ----------

def evaluate_model(model, df_test, image_name):
  # Evaluate the clasifier
  evaluation = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
  )
  
  df_pred = model.transform(df_test)
  pr_auc = evaluation.evaluate(df_pred)
  print("Test set PR AUC:", pr_auc)
  
  df_probs = (
    df_pred
    .select("label", "prediction", vector_to_array(F.col("probability")).getItem(1).alias("prob"))
    .toPandas()
  )

  # precision recall curve
  precision, recall, _ = precision_recall_curve(df_probs["label"], df_probs["prob"])
  f = plt.figure(figsize=(10,7))
  plt.plot(recall, precision, lw=2)
  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(loc="best")
  plt.title("precision vs. recall curve")
  plt.savefig(image_name)
  plt.show()
  
  return pr_auc

# COMMAND ----------

model_name = "german_credit"

# COMMAND ----------

# MAGIC %md
# MAGIC # Model A
# MAGIC This model will use the preprocessed dataframe, and use logistic regression to classify the credit requests.

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.spark.autolog(silent=False)
with mlflow.start_run(run_name="logistic_regression") as mlflow_run:
  lr = LogisticRegression(maxIter=1000)
  pipeline = Pipeline(stages=[str_indexer, hot_encoder, vector_assembler, str_indexer_label, lr])

  # Fit the pipeline to training documents.
  model = pipeline.fit(df_train)
  pr_auc = evaluate_model(model=model, df_test=df_test, image_name='/tmp/pr-curve-model-a.png')
  
  # Train on full dataset
  model = pipeline.fit(df_train.union(df_test))
  
  # Log on MLflow
  mlflow.spark.log_model(model, artifact_path="model")
  mlflow.log_metric(key="PR_AUC", value=pr_auc)
  mlflow.log_param(key="Stages", value=str(pipeline.getStages()))
  mlflow.log_param(key="MaxIter", value=lr.getMaxIter())
  mlflow.log_artifact("pr-curve-model-a.png")
  
  run_id = mlflow_run.info.run_id
  model_uri = f"runs:/{run_id}/model"
  model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model B
# MAGIC This model is similar to the previous one, but will use gradient boosted trees for the modelling. The data preprocessing remains the same

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.spark.autolog(silent=False)
with mlflow.start_run(run_name="gradient_boosted_trees") as mlflow_run:
  gbt = GBTClassifier()
  pipeline = Pipeline(stages=[str_indexer, hot_encoder, vector_assembler, str_indexer_label, gbt])

  # Fit the pipeline to training documents.
  model = pipeline.fit(df_train)
  pr_auc = evaluate_model(model=model, df_test=df_test, image_name='/tmp/pr-curve-model-b.png')
    
  # Train on full dataset
  model = pipeline.fit(df_train.union(df_test))
  
  # Log on MLflow
  mlflow.spark.log_model(model, artifact_path="model")
  mlflow.log_metric(key="PR_AUC", value=pr_auc)
  mlflow.log_param(key="Stages", value=str(pipeline.getStages()))
  mlflow.log_artifact("pr-curve-model-b.png")
  
  run_id = mlflow_run.info.run_id
  model_uri = f"runs:/{run_id}/model"
  model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------


