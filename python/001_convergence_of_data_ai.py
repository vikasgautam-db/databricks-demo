# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <h1 style="text-align: left; color: black">You need more than just good models !</h1>
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/challenges_in_AI.png" style="float: center; margin-left: 100px; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/dip_mosaic_overview_wb.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/solution_to_challenges.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/GenAI_architecture_patterns.png?" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Prompting** in the context of natural language processing (NLP) and large language models (LLMs) refers to the technique of crafting specific input prompts that guide the behavior of a language model to generate desired outputs. 
# MAGIC
# MAGIC - **Zero-Shot** (method of testing results on a base/foundation model without any examples and specific instructions)
# MAGIC - **Few-shot** (contains a few examples that the LLM takes in as input to learn and respond)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/elements_of_prompt.png" style="float: center; margin-right: 10px" width="900px" />
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

# MAGIC %sql 
# MAGIC
# MAGIC
# MAGIC select conversation, human_assigned_sentiment from vikas_demo.diday_mumbai.train_data 
# MAGIC where human_assigned_sentiment = 'Negative'
# MAGIC limit 1
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/databricks_prompting_architecture.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Let's take a look at an application built on this approach ###
# MAGIC
# MAGIC **Model Used:** QWEN-VL-Chat (a visual language model)
# MAGIC
# MAGIC **Source:** HuggingFace [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL) 
# MAGIC
# MAGIC **Serving:** Databricks Model Serving (AWS g5.4XLarge, GPU A10 Family)
# MAGIC
# MAGIC #### [Image Chat](https://2831939156970c1c3c.gradio.live/) ####

# COMMAND ----------

# MAGIC %md ### Let's dig into Fine Tuning... 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC Why Fine Tune:
# MAGIC
# MAGIC - Tailored models for specific tasks or domians
# MAGIC
# MAGIC - Understand Jargons, Acronyms and nomenclature that your company uses
# MAGIC
# MAGIC - Far Better accuracy
# MAGIC
# MAGIC - Competitive Advantage

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/fine_tuning_intro.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/fine_tuning_techniques_updated.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ### [Fine Tuning Example] ./python/003_PEFT_Finetune_eCommerce_summary
# MAGIC
# MAGIC
# MAGIC (https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/4145428256910587)
# MAGIC
# MAGIC Read this databricks [blog](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) for more details
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Considerations for Fine Tuning an AI Model ###
# MAGIC
# MAGIC </br>
# MAGIC </br>
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/87cfc98fa97cb9a05e7164649f5afa77f53a8c85/resources/di-day-mumbai/finetuning_scaling_options.png" style="float:center; margin-left: 100px" width="500px" />
# MAGIC

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
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/databricks_model_serving.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ### Retrieval Augmented generation
# MAGIC
# MAGIC
# MAGIC **Retrieval-Augmented Generation (RAG)** is an architectural approach that enhances the capabilities of large language models (LLMs) by integrating them with an information retrieval component. This architecture consists of two main parts: the retriever and the generator. The retriever uses techniques like vector similarity to scan through a knowledge base and retrieve the most relevant information based on the user's query. This information is then passed to the generator, typically a model like Llama or BART, which synthesizes the retrieved data into coherent, contextually relevant responses.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/vikasgautam-db/databricks-demo/main/resources/di-day-mumbai/rag-workflow.png" style="float: center; margin-right: 10px" width="900px" />
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md #### RAG Limitations ####
# MAGIC
# MAGIC - Iterative Reasoning Capabilities are missing
# MAGIC - Data Relevance and Retrieval Accuracy
