# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <h1 style="text-align: left; color: black">You need more than just good models !</h1>
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/challenges_in_AI.png" style="float: center; margin-left: 100px; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/dip_mosaic_overview_wb.png" style="float: center; margin-left: 100px; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/solution_to_challenges.png" style="float: center; margin-left: 100px; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/GenAI_architecture_patterns.png?" style="float: center; margin-left: 100px; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Prompt Engineering ##
# MAGIC
# MAGIC ### <u> **Prompting** </u> in the context of natural language processing (NLP) and large language models (LLMs) refers to the technique of crafting specific input prompts that guide the behavior of a language model to generate desired outputs. ###
# MAGIC
# MAGIC ### - <u> Zero-Shot </u> (method of testing results on a base/foundation model without any examples and specific instructions)
# MAGIC ### - <u> Few-shot </u> (contains a few examples that the LLM takes in as input to learn and respond)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/elements_of_prompt.png" style="float: center; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ### Prompting with Databricks Playground ###
# MAGIC
# MAGIC
# MAGIC You can find the complete dataset on HuggingFace [NebulaByte/E-Commerce_Customer_Support_Conversations](https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/databricks_prompting_architecture.png" style="float: center; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/fine_tuning_intro.png" style="float: center; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Let's take a look at a few applications built on models trained with this approach ###
# MAGIC
# MAGIC #### [Motor Claims](https://16134882ef74ed6cea.gradio.live/) ####
# MAGIC
# MAGIC **Base Model Used:** QWEN-VL-Chat (a visual language model)
# MAGIC
# MAGIC **Source:** HuggingFace [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL) 
# MAGIC
# MAGIC **Serving:** Databricks Model Serving (AWS g5.4XLarge, GPU A10 Family)
# MAGIC
# MAGIC
# MAGIC #### [eCommerce Q & A](https://6f7b3593a43624ef1b.gradio.live/) ####
# MAGIC **Base Model Used:** Google FLAN T5 Base 
# MAGIC
# MAGIC **Source:** HuggingFace [goolge/flan-t5-base](https://huggingface.co/google/flan-t5-base) 
# MAGIC
# MAGIC **Serving:** Databricks Model Serving (AWS g5.4XLarge, GPU A10 Family)
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Training is very simple on Databricks Data Intelligence platform MLFlow ##
# MAGIC
# MAGIC ##[Example](https://e2-dogfood.staging.cloud.databricks.com/ml/experiments?o=6051921418418893)##

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Models in Unity Catalog ###
# MAGIC
# MAGIC ### [Models in Catalog](https://e2-dogfood.staging.cloud.databricks.com/explore/data/main/vikas_demo?o=6051921418418893)###

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/pretraining.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC
# MAGIC #### - - [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) - - [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) - - [BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) - - [FinBERT](https://huggingface.co/ProsusAI/finbert) ####

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/databricks_model_serving.png" style="float: center; margin-right: 10px" width="1300px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # Solution Accelerator : Large Language Models (LLMs) for Customer Service Analytics #
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/analyze_customer.png" style="float: Left; margin-right: 10px" width="600px" />
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/qrcode_www.databricks.com.png" style="float: right; margin-right: 10px" width="600px" />

# COMMAND ----------


