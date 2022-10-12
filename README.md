# AB Testing Accelerator
In any machine learning related project, training a model offline is just one part of the process. In 2020, we saw how the whole world quickly changed due to the pandemic. When working with data that represents the outside world is it important to keep in mind that models are going to have different accuracies over time because the data used for that training might no longer be representative, also known as model drift. Hence, it is important to track the real world accuracy over time. 

Moreover, training offline new models might or might not lead to better real world performance. This is why A/B testing can be a good technique to understand the effects of making changes to the systems that consume these machine learning models, and help us making data driven decisions.

<img src="https://ml-ops.org/img/mlops-loop-en.jpg" width="400"/>

This series of notebooks we will demostrate the following:
- How to do online inference in real time using Structured Streaming
- How to do A/B testing with two machine learning models registered with MLflow
- Detect model drift over time and visualize it with Databricks SQL

We will use a toy dataset related to credit risk. See the next cell for more details.


The system that we will setup is the following:

<img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_1.png?raw=true" width="1000"/>

With this system we will:
- Take credit risk data and trains two machine learning models with it. The models will predict the risk of giving a credit requested by a person.
- The models will be registered with MLflow.
- Create a live stream of new credit requests. We will use a Delta table, although this system would be compatible with other technologies such as Kafka. These requests will come from the credit risk dataset for demostration purposes.
- Load the two trained ML models, and we will make real time predictions on new credit requests. The predictions will be saved as a Delta table (also streaming), although we give a suggestion on how we could deliver them to a Kafka server to export them to other systems.
- We assume that there is a feedback loop, where we collect new grounth truth data related to the requests for which we made predictions. This means that we collect information about if the people who requested a credit actually paid back. For the sake of this exercise we will use again the credit risk dataset.
- This feedback loop will be used to compare over time the predictions with the actual responses from the real world on both models. We will visualize on Databricks SQL how both models perform, effectivelly doing A/B testing and model drift all in one.

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
