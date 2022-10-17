# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if another user alters the workflow and cluster via UI, running this script again resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 14400,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG"
        },
        "tasks": [
            {
                "job_cluster_key": "ab_testing_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"1. Introduction"
                },
                "task_key": "ab_testing_01",
                "description": ""
            },
            {
                "job_cluster_key": "ab_testing_cluster",
                "notebook_task": {
                    "notebook_path": f"2. Model training"
                },
                "task_key": "ab_testing_02",
                "depends_on": [
                    {
                        "task_key": "ab_testing_01"
                    }
                ]
            },
            {
                "job_cluster_key": "ab_testing_cluster",
                "notebook_task": {
                    "notebook_path": f"3. Start streaming sources"
                },
                "task_key": "ab_testing_03",
                "depends_on": [
                    {
                        "task_key": "ab_testing_02"
                    }
                ]
            },
            {
                "job_cluster_key": "ab_testing_cluster",
                "notebook_task": {
                    "notebook_path": f"4. Real time inference"
                },
                "task_key": "ab_testing_04",
                "depends_on": [
                    {
                        "task_key": "ab_testing_02"
                    }
                ]
            },
            {
                "job_cluster_key": "ab_testing_cluster",
                "notebook_task": {
                    "notebook_path": f"5. AB testing metrics"
                },
                "task_key": "ab_testing_05",
                "depends_on": [
                    {
                        "task_key": "ab_testing_04"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "ab_testing_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.sql.streaming.stopTimeout": "60000"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------



# COMMAND ----------


