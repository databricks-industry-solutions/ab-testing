# Databricks notebook source
# MAGIC %md 
# MAGIC # How to run this ab testing accelerator:
# MAGIC 
# MAGIC ### Method 1: 
# MAGIC 
# MAGIC You may run just this notebook, since it sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. You can check Workflow job runs link at the end of this notebook to see how this solution accelerator executes. 
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if another user alters the workflow and cluster via UI, running this script again resets them.
# MAGIC 
# MAGIC ### Method 2: 
# MAGIC 
# MAGIC You may directly set up your own 10.4+ ML cluster, and run notebooks 1-5 sequentially and interactively. You may delete some of the "time.sleep()" cells in order to save some time (since these are mostly designed for enough time gaps between notebooks runs for the workflow set up), as long as you ensure having enough gaps among running notebook 3-5 after notebook1&2. 
# MAGIC 
# MAGIC ### Happy exploring!
# MAGIC 
# MAGIC 
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
                        "task_key": "ab_testing_02"
                    }
                ]
            },
            {
                "job_cluster_key": "ab_testing_cluster",
                "notebook_task": {
                    "notebook_path": f"5. AB testing metrics"
                },
                "task_key": "ab_testing_06",
                "depends_on": [
                    {
                        "task_key": "ab_testing_05"
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

# MAGIC %md
# MAGIC # How to check the results:
# MAGIC 
# MAGIC * You may check the automatically created jobs under "Workflows" tab->"Job runs" tab; or you may edit to get the link by copy paste the job information printed below directly following workspace URL. For example, "Job #958240233939092-649491 is RUNNING" corresponds to replacing the current page URL starting from "#job/...", so it will look like ```<DB-workspace-url>#job/958240233939092/run/649491``` "TODO"
# MAGIC * One trigger of the run will take about 31 minutes in total. Users can modify some of the "sleep" timers for their own use cases. 
# MAGIC * If one triggers a new run too soon from this same notebook, an error of "SKIPPED" may show up. Please finish your first Job runs before triggering a new one. 
# MAGIC * Here is an example of the graph: "TODO"

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "True", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)
