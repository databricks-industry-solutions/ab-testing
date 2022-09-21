# Databricks notebook source
# MAGIC %md ---
# MAGIC title: A/B testing with MLflow 1 - Introduction
# MAGIC authors:
# MAGIC -  Sergio Ballesteros
# MAGIC tags:
# MAGIC - machine-learning
# MAGIC - python
# MAGIC - pyspark
# MAGIC - a/b testing
# MAGIC - ab testing
# MAGIC - binary-classifier
# MAGIC - mllib
# MAGIC - credit risk
# MAGIC - loan risk
# MAGIC - finance
# MAGIC created_at: 2021-07-27
# MAGIC updated_at: 2021-07-27
# MAGIC tldr: Introduction to the A/B testing series of notebooks. This series of notebooks shows how to leverage Databricks with MLflow and Delta to do A/B testing on streaming data related to credit risk
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/10781635/](https://demo.cloud.databricks.com/#notebook/10781635/)

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
# MAGIC In any machine learning related project, training a model offline is just one part of the process. In 2020, we saw how the whole world quickly changed due to the pandemic. When working with data that represents the outside world is it important to keep in mind that models are going to have different accuracies over time because the data used for that training might no longer be representative, also known as model drift. Hence, it is important to track the real world accuracy over time. 
# MAGIC 
# MAGIC Moreover, training offline new models might or might not lead to better real world performance. This is why A/B testing can be a good technique to understand the effects of making changes to the systems that consume these machine learning models, and help us making data driven decisions.
# MAGIC 
# MAGIC <img src="https://ml-ops.org/img/mlops-loop-en.jpg" width="400"/>
# MAGIC 
# MAGIC This series of notebooks we will demostrate the following:
# MAGIC - How to do online inference in real time using Structured Streaming
# MAGIC - How to do A/B testing with two machine learning models registered with MLflow
# MAGIC - Detect model drift over time and visualize it with Databricks SQL
# MAGIC 
# MAGIC We will use a toy dataset related to credit risk. See the next cell for more details.
# MAGIC 
# MAGIC 
# MAGIC The system that we will setup is the following:
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_1.png?raw=true" width="1000"/>
# MAGIC 
# MAGIC With this system we will:
# MAGIC - Take credit risk data and trains two machine learning models with it. The models will predict the risk of giving a credit requested by a person.
# MAGIC - The models will be registered with MLflow.
# MAGIC - Create a live stream of new credit requests. We will use a Delta table, although this system would be compatible with other technologies such as Kafka. These requests will come from the credit risk dataset for demostration purposes.
# MAGIC - Load the two trained ML models, and we will make real time predictions on new credit requests. The predictions will be saved as a Delta table (also streaming), although we give a suggestion on how we could deliver them to a Kafka server to export them to other systems.
# MAGIC - We assume that there is a feedback loop, where we collect new grounth truth data related to the requests for which we made predictions. This means that we collect information about if the people who requested a credit actually paid back. For the sake of this exercise we will use again the credit risk dataset.
# MAGIC - This feedback loop will be used to compare over time the predictions with the actual responses from the real world on both models. We will visualize on Databricks SQL how both models perform, effectivelly doing A/B testing and model drift all in one.

# COMMAND ----------

# MAGIC %md
# MAGIC # Download the dataset
# MAGIC <img src="https://thumbs.dreamstime.com/b/credit-risk-message-bubble-word-cloud-collage-business-concept-background-credit-risk-message-bubble-word-cloud-collage-business-216251701.jpg" width="600"/>
# MAGIC 
# MAGIC Our toy dataset will be the German Credit Risk dataset:
# MAGIC 
# MAGIC https://www.kaggle.com/uciml/german-credit
# MAGIC 
# MAGIC A preview of the data is available here https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/german_credit_data.csv
# MAGIC 
# MAGIC We will save the data in the *german_credit_data* Delta table

# COMMAND ----------

# OPTION1: download and that csv file and upload it to DBFS into the path as shown below, under databricks-datasets-private folder, then under ML folder etc.
# permanent_table_name = "german_credit_data"

# df = (
#   spark
#   .read
#   .option("inferSchema", "true") 
#   .option("header", "true") 
#   .option("sep", ",") 
#   .csv("/mnt/databricks-datasets-private/ML/credit-risk/german_credit_data.csv") 
# )

# df.write.format("delta").mode("overwrite").saveAsTable(permanent_table_name)

# COMMAND ----------

# DBTITLE 1,Option2: manually upload this csv data into your default database, with table name "german_credit_data"
permanent_table_name = "german_credit_data"
display(spark.read.table(permanent_table_name))

# COMMAND ----------


