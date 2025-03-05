# %%
import pandas as pd
import os
import requests
import json
import numpy as np
from tqdm import tqdm
import ollama

# %%
def Get_Respond_LLM(prompt, model):
    response = ollama.chat(model, messages=[{
        'role': 'user',
        'content': prompt,
    }])

    return response['message']['content']


def Get_Respond_LLM_Local(prompt, Model):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": Model,
        "messages": [
            {
                "role": "user",
                "content": prompt

            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]


# %%
def Clean_extra(payload, comma, line, space):
    # a function to process the  LLMs output
    payload = payload.replace(".", "")
    payload = payload.replace('"', "")
    payload = payload.replace("'", "")
    payload = payload.replace("-", "")
    payload = payload.replace("and ", "")
    if space:
        payload = payload.replace(" ", "")
    if comma:
        payload = payload.replace(",", "")
    if line:
        payload = payload.replace("\\n", "")
    return payload


def Process_function(string, debug):
    # this function cleans up the ourtput of the LLM
    splits = string.split('---')
    seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
    seed = Clean_extra(seed, True, True, True) # the original seed name
    payload = splits[-1]
    
    # the case that we have nice comma seperated 
    if len(payload.split(","))>10:
        payload = Clean_extra(payload, False, True, True)
        for num in range(10):
            payload = payload.replace(str(num), "")
        name_list = list(payload.split(",")[1:11])
        # remove repated setance, and hallucinations  
        Cleaned_name_list = []
        for name in name_list:
            # if we have: it is the case of Here are 10 alternative spellings for the name Rowena: Rowenna",
            if ":" in name:
                c_name = name.split(":")[-1]
                Cleaned_name_list.append(c_name)
            elif len(name) > 2*len(seed):
                # error in processing
                Cleaned_name_list.append(np.nan)
            else:
                Cleaned_name_list.append(name)
                
        if len(Cleaned_name_list) == 10:        
            if debug:
                return seed, "r1", Cleaned_name_list, payload
            else:
                return seed, "r1", Cleaned_name_list
    else:
        # the case that we dont have comma seperated
        # how many lines do I have in this respond
        len_ans = len(payload.split("\\n"))
        if len_ans >2: # multiple lines, I will use this to seprate the names
            payload = Clean_extra(payload, True, False, True)
            for num in range(10): #llm itarates with number instead of comma
                payload = payload.replace(str(num), "")
            name_list = list(payload.split("\\n"))[0:10]
            Cleaned_name_list = []
            for name in name_list:
                    # if we have: it is the case of Here are 10 alternative spellings for the name Rowena: Rowenna",
                    if ":" in name:
                        c_name = name.split(":")[-1]
                        Cleaned_name_list.append(c_name)
                    elif len(name) > 2*len(seed):
                        # error in processing
                        Cleaned_name_list.append(np.nan)
                    else:
                        Cleaned_name_list.append(name)
            if debug:
                return seed, "r2", Cleaned_name_list, payload
            else:
                return seed, "r2", Cleaned_name_list
        else: 
        # just itirated withouth the comma and only by numbers, so I will use empty space as a sep
        #' 1. Lilac  2. Lillie  3. Lyra  4. Lily-Jayne  5. Lis-Lis  6. Liliana  7. Lilis  8. Lillian  9. Lylla  10. Lylyla\\n',
            payload = Clean_extra(payload, True, True, False)
            for num in range(10): #llm itarates with number instead of comma
                payload = payload.replace(str(num), "")
                
            name_list = list(payload.split(" ")) # we get more than 10
            
            Cleaned_name_list = []
            for name in name_list:
                    # if we have: it is the case of Here are 10 alternative spellings for the name Rowena: Rowenna",
                    if ":" in name:
                        c_name = name.split(":")[-1]
                        Cleaned_name_list.append(c_name)
                    elif len(name) > 2*len(seed):
                        # error in processing
                        Cleaned_name_list.append(np.nan)
                    elif len(name) != 0:
                        Cleaned_name_list.append(name)
            if debug:
                return seed, "r3", Cleaned_name_list, payload
            else:
                return seed, "r3", Cleaned_name_list

# %%
def encode_names_to_json(names_list, query, output_path, filename="names_to_process.json"):
    """
    Creates a JSON file from a list of names and a query to be processed later.
    
    Args:
        names_list (list): List of names to generate variations for
        query (str): The query template to use with each name
        output_path (str): Directory to save the JSON file
        filename (str, optional): Name of the output file
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create the data structure
    data = {
        "names": names_list,
        "query": query
    }
    
    # Save to JSON file
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    print(f"JSON file with names and query saved to: {os.path.abspath(output_file)}")
    return output_file


def decode_json_to_names_and_query(json_path):
    """
    Decodes a JSON file into a list of names and a query.
    
    Args:
        json_path (str): Path to the JSON file
    
    Returns:
        tuple: (list of names, query string)
    """
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the list of names and the query
    names_list = data["names"]
    query = data["query"]
    
    return names_list, query


# %%


def process_variations_to_json(Response_list, output_path, model_name, Process_function):
    """
    Processes LLM responses to extract name variations and saves them to JSON.
    
    Args:
        Response_list (list): List of LLM responses
        output_path (str): Directory to save the JSON file
        model_name (str): Name of the model used (for filename)
        Process_function (function): Function to process LLM responses
    
    Returns:
        str: Path to the saved JSON file
    """
    # Split the responses
    Responds = "".join(Response_list).split("Respond")
    
    # First, create a dictionary to store each name and its variations
    name_variations = {}
    max_variations = 0

    # Process each response to extract variations
    for i in range(1, len(Responds)):
        try:
            llm_respond = Process_function(Responds[i], False)
            name = llm_respond[0]
            variations = [var for var in llm_respond[2] if not pd.isna(var) and var != ""]
            name_variations[name] = variations
            
            # Track the maximum number of variations
            max_variations = max(max_variations, len(variations))
        except Exception as e:
            print(f"Error processing response {i}: {e}")

    # Create a new DataFrame with columns for the maximum number of variations
    columns = ['Name'] + [f'Var_{i+1}' for i in range(max_variations)]
    result_df = pd.DataFrame(columns=columns)

    # Fill the DataFrame with names and their variations
    for i, (name, variations) in enumerate(name_variations.items()):
        # Create a row with the name and its variations, padding with empty strings if needed
        row_data = [name] + variations + [''] * (max_variations - len(variations))
        result_df.loc[i] = row_data

    # Clean the data - convert to string and remove unwanted characters
    for r in range(result_df.shape[0]):
        input_row = result_df.iloc[r,:]
        input_row = input_row.astype(str).apply(lambda x: x.replace(")", ""))
        input_row = input_row.astype(str).apply(lambda x: x.replace("(", ""))
        input_row = input_row.astype(str).apply(lambda x: x.replace("]", ""))
        input_row = input_row.astype(str).apply(lambda x: x.replace("[", ""))
        input_row = input_row.astype(str).apply(lambda x: x.replace(",", ""))
        result_df.iloc[r,:] = input_row

    # Save DataFrame to pickle for backup
    result_df.to_pickle(os.path.join(output_path, f"{model_name}_df.pkl"))

    # Save to JSON
    json_data = {}
    for i, row in result_df.iterrows():
        name = row['Name']
        variations = [var for var in row[1:] if var != ""]
        json_data[name] = variations

    # Save to JSON file
    json_path = os.path.join(output_path, f"{model_name}_names.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f"JSON file with processed variations saved to: {json_path}")
    return json_path

# %%
# list of availabe models. if a model is not avaialbe you can simply add by using ollama pull, 
# here is the list of availabe models. https://ollama.com/library?sort=popular

# %%
ollama.list()["models"]

# %%
## validation

# %%
# List of names to process
seed_names = ["gillbert", "jamehriah", "joana", "wynnfred", "Camille"]

# Query template
query = "Give me 10 comma sperated alternative spellings of the name {name}. 5 of them should be sound similar to the original name and 5 should be orthographic similar. provide only the names"

# Output path
output_path = "/Users/asemothman/repo/YEGM/Test/"

# Encode names and query to JSON
json_file = encode_names_to_json(
    names_list=seed_names,
    query=query,
    output_path=output_path,
    filename="names_to_process.json"
)

print(f"Names and query encoded to: {json_file}")

# %%
### mining 

# %%
# Path to the JSON file from validator
json_path = "/Users/asemothman/repo/YEGM/Test/names_to_process.json"

# Output path for processed data
output_path = "/Users/asemothman/repo/YEGM/Test/"

# Model to use
Model = "tinyllama:latest"

# Decode the JSON to get names and query
names_list, query_template = decode_json_to_names_and_query(json_path)

print(names_list)
print(query_template)

# %%

# Generate variations using LLM
Response_list = []
for name in tqdm(names_list):
    Response_list.append("Respond")
    Response_list.append("---")
    Response_list.append("Query-" + name)
    Response_list.append("---")
    
    # Format the query with the current name
    formatted_query = query_template.replace("{name}", name)
    
    # Query the LLM
    name_respond = Get_Respond_LLM(formatted_query, Model)
    Response_list.append(name_respond)

# Save raw responses to file (optional)
with open(os.path.join(output_path, f"Mining_mined_{Model.replace(':', '_')}.txt"), 'wt', encoding='utf-8') as f:
    f.write(str(Response_list))

# Process variations and save to JSON
result_json = process_variations_to_json(
    Response_list=Response_list,
    output_path=output_path,
    model_name=Model.replace(':', '_'),
    Process_function=Process_function  # Your existing function to process responses
)

print(f"Processed variations saved to: {result_json}")

# %%



