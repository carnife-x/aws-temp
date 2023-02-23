#!/usr/bin/env python
# coding: utf-8

from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import boto3

role = sagemaker.get_execution_role()


hub = {
  'HF_MODEL_ID':'EleutherAI/gpt-neo-1.3B', # model_id from hf.co/models
  'HF_TASK':'text-generation' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
   py_version="py36", # python version of the DLC
)


session = boto3.session.Session()
region = session.region_name
huggingface_model.prepare_container_def("ml.t3.medium")


predictor = huggingface_model.deploy(
    endpoint_name="endpoint-hf-test",
   initial_instance_count=1,
   instance_type="ml.c5d.2xlarge",
)

# predictor.predict({
# 	"inputs": "Write a short, thought-provoking, and eye-catching post title for Hacker News (https://news.ycombinator.com/) submission based on a blog post’s provided description. \n Post description: The post describes how to deliver constructive feedback in difficult situations. The author claims that thoughtful, empathetic language can make or break business relationships and suggests that excellent communication is not just about what you say; it is about what other people hear. \n Post title: A guide to difficult conversations \n Post description: The post describes applying the behavior model invented by Stanford professor BJ Fogg to battle procrastination. The author suggests the three things we need to change the behavior and explains why we procrastinate. \n Post title: How to stop procrastinating by using the Fogg Behavior Model \n Post description:  The post explains that conventional education does not teach students how to use tools like the command line, text editors, and control systems. The idea is that programmers spend hundreds of hours using these tools and therefore need to master them to work more efficiently. \n Post title: The missing semester of CS education \n Post description: The post describes the process for taking thoughtful meeting notes. The author covers how he trained himself to remember what is happening during long meetings and suggests actionable steps to integrate the note-taking habit in your life. \n Post title:",
#     "parameters": {
#         "max_length": 300,
#         "temperature":0.8,
#         "return_full_text": False,
#     }
# })




