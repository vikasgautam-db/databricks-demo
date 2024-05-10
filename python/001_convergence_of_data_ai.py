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

# MAGIC %md-sandbox ### Prompting with Databricks Playground ###

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Categorize the sentiment of the customer in the below conversation into one of Neutral, Frustrated, Positive or Negative - output only the sentiment and nothing else.
# MAGIC
# MAGIC #### Conversation::
# MAGIC
# MAGIC Agent: Thank you for calling BrownBox Customer Support. My name is Alex. How may I assist you today?
# MAGIC
# MAGIC Customer: Hi, Alex. I need help with a refund for my Cash on Delivery payment.
# MAGIC
# MAGIC Agent: I'm sorry to hear that. Can you please provide me with your order details?
# MAGIC
# MAGIC Customer: Yes, my order number is BB12345.
# MAGIC
# MAGIC Agent: Thank you. May I know the reason for the refund?
# MAGIC
# MAGIC Customer: I received a defective Electric Cooker, and I want to return it.
# MAGIC
# MAGIC Agent: I understand. We apologize for the inconvenience caused. We will process your refund for the Cash on Delivery payment. However, we cannot reimburse the courier charges for the return of the Electric Cooker.
# MAGIC
# MAGIC Customer: What? That's not fair. The Electric Cooker was defective, and it's not my fault.
# MAGIC
# MAGIC Agent: I understand your concern, but our policy states that we do not reimburse courier charges for returns. However, we can offer you a discount on your next purchase as a gesture of goodwill.
# MAGIC
# MAGIC Customer: I don't want a discount. I want my courier charges to be reimbursed.
# MAGIC
# MAGIC Agent: I'm sorry, but that's not possible. Is there anything else I can assist you with?
# MAGIC
# MAGIC Customer: No, I'm not satisfied with your response. Can you transfer me to a more experienced agent?
# MAGIC
# MAGIC Agent: I understand your concern, but I assure you that I am capable of handling your issue. However, if you still want to speak to a more experienced agent, I can transfer your call. Please bear with me for a moment.
# MAGIC
# MAGIC [Agent puts the customer on hold to transfer the call]
# MAGIC
# MAGIC Agent: Thank you for your patience. I have transferred your call to my senior colleague, who will be able to assist you further. Please stay on the line.
# MAGIC
# MAGIC [Senior agent picks up the call]
# MAGIC
# MAGIC Senior Agent: Hello, this is David, a senior agent from BrownBox Customer Support. How may I assist you today?
# MAGIC
# MAGIC Customer: Hi, David. I need help with a refund for my Cash on Delivery payment and reimbursement of courier charges for a defective Electric Cooker.
# MAGIC
# MAGIC Senior Agent: I'm sorry to hear that. Can you please provide me with your order details?
# MAGIC
# MAGIC Customer: Yes, my order number is BB12345.
# MAGIC
# MAGIC Senior Agent: Thank you. I understand your concern regarding the reimbursement of courier charges. Let me check if we can make an exception in your case. Please bear with me for a moment.
# MAGIC
# MAGIC [Senior agent puts the customer on hold to check the policy]
# MAGIC
# MAGIC Senior Agent: Thank you for your patience. I have checked with my team, and we can make an exception in your case. We will process your refund for the Cash on Delivery payment and reimburse the courier charges for the return of the Electric Cooker.
# MAGIC
# MAGIC Customer: Thank you, David. I appreciate your help.
# MAGIC
# MAGIC Senior Agent: You're welcome! Is there anything else I can assist you with today?
# MAGIC
# MAGIC Customer: No, that's all for now.
# MAGIC
# MAGIC Senior Agent: Alright. Thank you for choosing BrownBox, and have a great day!
# MAGIC
# MAGIC Customer: You too. Goodbye!
# MAGIC
# MAGIC Senior Agent: Goodbye!
# MAGIC
# MAGIC ####Sentiment:: 
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
# MAGIC #### [Motor Claims](https://a1530e52339aad2e13.gradio.live/) ####
# MAGIC
# MAGIC **Base Model Used:** QWEN-VL-Chat (a visual language model)
# MAGIC
# MAGIC **Source:** HuggingFace [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL) 
# MAGIC
# MAGIC **Serving:** Databricks Model Serving (AWS g5.4XLarge, GPU A10 Family)
# MAGIC
# MAGIC
# MAGIC #### [eCommerce Q & A](https://4f49459d7f70d132ae.gradio.live/) ####
# MAGIC **Base Model Used:** FLAN T5 Base 
# MAGIC
# MAGIC **Source:** HuggingFace [flan-t5-base](https://huggingface.co/google/flan-t5-base) 
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


