# Databricks notebook source
# MAGIC %md This notebook series is also available at https://github.com/databricks-industry-solutions/ab-testing.

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
# MAGIC In any machine learning related project, training a model offline is just one part of the process. In 2020, we saw how the whole world quickly changed due to the pandemic. When working with data that represents the outside world is it important to keep in mind that models are going to have different accuracies over time because the data used for that training might no longer be representative, also known as model drift. Hence, it is important to track the real world accuracy over time. 
# MAGIC 
# MAGIC Moreover, training offline new models might or might not lead to better real world performance. This is why A/B testing can be a good technique to understand the effects of making changes to the systems that consume these machine learning models, and help us making data driven decisions.
# MAGIC 
# MAGIC <img src="https://ml-ops.org/img/mlops-loop-en.jpg" width="400"/>
# MAGIC 
# MAGIC In this series of notebooks, we will demostrate the following:
# MAGIC - How to do online inference in real time using Structured Streaming
# MAGIC - How to do A/B testing with two machine learning models registered with MLflow
# MAGIC - Detect model drift over time and visualize it with Databricks SQL Dashboard
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

# MAGIC %sh
# MAGIC wget "https://raw.githubusercontent.com/sergioballesterossolanas/databricks-ab-testing/master/german_credit_data.csv" -O /dbfs/tmp/german_credit_data.csv

# COMMAND ----------

# MAGIC %sql 
# MAGIC drop database if exists solacc_ab_test cascade;
# MAGIC create database solacc_ab_test;
# MAGIC use solacc_ab_test;

# COMMAND ----------

permanent_table_name = "solacc_ab_test.german_credit_data"

df = (
  spark
  .read
  .option("inferSchema", "true") 
  .option("header", "true") 
  .option("sep", ",") 
  .csv("/tmp/german_credit_data.csv")
)

df.write.format("delta").mode("overwrite").saveAsTable(permanent_table_name)

# COMMAND ----------

display(df)
