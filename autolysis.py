# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "chardet",
#   "python-dotenv"
#  ]
# ///

import pandas as pd
import requests
import json
import chardet
# import statsmodels.api as sm
import seaborn as sns
import sys
import os
import traceback
from dotenv import load_dotenv
import matplotlib.pyplot as plt


load_dotenv()


api_key = os.getenv("AIPROXY_TOKEN")



def Get_encoding(file_path):
    # Step 1: Detect file encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()  # Read the raw bytes of the file
        result = chardet.detect(raw_data)  # Detect encoding
        encoding = result['encoding']  # Extract the detected encoding
        return(encoding)


def Get_data(file_path):
    try:
        with open(file=file_path, mode='r', encoding=encoding) as f:
            first_ten= ''.join([f.readline () for i in range(25)])
            return first_ten
    except Exception as e:
        print(f"Error while reading the CSV: {e}")
        return None

file_path = sys.argv[1]
encoding = Get_encoding(file_path)
data = Get_data(file_path)

print(data)

url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
model= "gpt-4o-mini"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"}

content = "given the csv data where first line is heading, identify column name and datatype of that column, ignore missing values, if confuse use majority to decide datatype that a column contains."

functions=[
  {
    "name":"extract_column_meta_data",
    "description":"identify column name and their datatype",
    "parameters":{
      "type": "object",
      "properties": {
        "column_metadata": {
          "type": "array",
          "description": "metadata about each column",
          "items": {
            "type": "object",
            "properties": {
              "column_name": {
                "type": "string",
                "description": "name of the column"
              },
              "column_type": {
                "type": "string",
                "description": "The data type of column (e.g., string, integer)"
              }
            },
            "required": ["column_name", "column_type"]
          },
          "minItems": 1
        }
      },
      "required": ["column_metadata"]
    }

  }
]

json_data = {
    "model":model,
    "messages":[
        {"role":"system", "content":content},
        {"role":"user", "content":data}
    ],
    "functions": functions,
    "function_call": {"name":"extract_column_meta_data"}
}

r = requests.post(url=url, headers=headers, json=json_data)

print(r.status_code)

r.json()

Metadata= json.loads(r.json()["choices"][0]["message"]["function_call"]["arguments"])['column_metadata']

# Metadata

# print(type(Metadata))

df= pd.read_csv(file_path, encoding= encoding)

# column_names_list = df.columns.tolist()
# print(column_names_list)

# unique_count = {}
# for column in column_names_list:
#     unique_count[column] = len(df[column].unique())

# print(unique_count)

# column_stats =[]
# for column in column_names_list:
#   column_stats.append((df[column].describe()))

# column_stats

missing_values_summary = df.isnull().sum()
# Convert the summary to a DataFrame for plotting and sort by missing values
missing_values_df = missing_values_summary.reset_index()
missing_values_df.columns = ['Column', 'Missing Values']
missing_values_df = missing_values_df.sort_values(by='Missing Values')

# Plot the missing values summary
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=missing_values_df, x='Column', y='Missing Values')
plt.title('Missing Values Summary')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')

# Adjust x labels to ensure alignment
ax.set_xticks(range(len(missing_values_df)))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adding the annotations
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10), textcoords='offset points')

# Save the plot as a PNG file
plt.savefig('Missing_values.png')

df = df.dropna()
functions_1=[
  {
    "name":"provide_suggestion",
    "description":"given metadata of columns of a csv provide suggestion about various analysis that can be performed",
    "parameters":{
      "type": "object",
      "properties": {
        "suggest_analysis": {
          "type": "array",
          "description": "suggest suitable data analysis",
          "items": {
            "type": "object",
            "properties": {
              "suggested_analysis": {
                "type": "string",
                "description": "name of the analysis(eg. regression,correlation, clustering etcs)"
              },
              "reason": {
                "type": "string",
                "description": "reason, why are you suggesting this analysis"
              }
            },
            "required": ["suggested_analysis", "reason"]
          },
          "minItems": 1
        }
      },
      "required": ["suggest_analysis"]
    }

  }
]

Prompt = "Given metadata about columns, suggest various analysis that can be done also give reason about it"

def suggestions(Prompt, Metadata):
  url= "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
  headers = {"Authorization": f"Bearer {api_key}"}
  info_data = {
    "model":"gpt-4o-mini",
    "messages":[
        {"role":"system", "content":Prompt},
        {"role":"user", "content":json.dumps(Metadata)}
    ],
    "functions": functions_1,
    "function_call": {"name":"provide_suggestion"}
    }
  r= requests.post(url=url, headers=headers, json=info_data)
  result= r.json()
  return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])['suggest_analysis']

suggestion = suggestions(Prompt, Metadata)


functions_2=[
  {
    "name":"provide_code",
    "description":"given metadata of columns and suggestion, write compact concise python code",
    "parameters":{
      "type": "object",
      "properties": {
        "write_code": {
          "type": "array",
          "description": "write python code without comments",
          "items": {
            "type": "object",
            "properties": {
              "code0": {
                "type": "string",
                "description": "write python code to 'generate 'png' using seaborn and matplotlib only"
              },
              "code1": {
                "type": "string",
                "description": "write python code to 'generate png' using seaborn and matplotlib only"
              },
              "code2": {
                "type": "string",
                "description": "write python code to 'generate png' using seaborn and matplotlib only"
              },
            },
            "required": ["code0","code1","code2"]
          },
          "minItems": 1
        }
      },
      "required": ["write_code"]
    }

  }
]

Instruction = ("given metadata about columns of csv and suggestion about analysis"
"write compact & concise python code one after another without any comment to generate ''3' different pngs'"
"'only use seaborn & Matplotlib' to generate ''three' different pngs' that represent some key visual insight about data"
"please generate simple, easy to understand, elegant and informative pngs(charts)"
"don't write any comments in code"
"don't generate your own data, I have data as '''df''' from same csv for which metadata and suggestion are given"
"please do check if the parameter is depricated before using any parameter, don't use depricated parameter"
"your job is to do things as per instructions, work diligently"
"please don't violate any instruction, 'these are very 'crucial''")

def Generate_code(Instruction, Metadata, suggestion):
  url= "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
  headers = {"Authorization": f"Bearer {api_key}"}
  info_data = {
    "model":"gpt-4o-mini",
    "messages":[
        {"role":"system", "content":Instruction},
        {"role":"user", "content":json.dumps(Metadata)},
        {"role":"user", "content":json.dumps(suggestion)}
    ],
    "functions": functions_2,
    "function_call": {"name":"provide_code"}
    }
  r= requests.post(url=url, headers=headers, json=info_data)
  result= r.json()
  return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])['write_code']

code = Generate_code(Instruction, Metadata, suggestion)
print(code)
# code

# clean_code = code.strip("'")
# clean_code

# len(code)

# code[0]['code0']

# Execute code blocks using a for loop
for i in range(len(code)):
    try:
        key = f"code{i}"
        exec(code[i][key])
    except Exception as e:
        traceback.print_exc()

# try:
#     # Code that might raise an exception
#     exec(code)
# except Exception as e:
#     # Code to handle the exception
#     print(e)

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# sns.scatterplot(data=df, x='longitude', y='latitude', hue='business_id', palette='viridis')
# plt.title('Geospatial Distribution of Businesses')
# plt.savefig('geospatial_distribution.png')
# plt.close()

# df

# # Convert the summary to a DataFrame for plotting and sort by missing values
# missing_values_df = missing_values_summary.reset_index()
# missing_values_df.columns = ['Column', 'Missing Values']
# missing_values_df = missing_values_df.sort_values(by='Missing Values')

# # Plot the missing values summary
# plt.figure(figsize=(12, 6))
# ax = sns.barplot(data=missing_values_df, x='Column', y='Missing Values')
# plt.title('Missing Values Summary')
# plt.xlabel('Columns')
# plt.ylabel('Number of Missing Values')

# # Adjust x labels to ensure alignment
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# # Adding the annotations
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.0f'),
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center',
#                 xytext = (0, 10), textcoords = 'offset points')

# plt.show()