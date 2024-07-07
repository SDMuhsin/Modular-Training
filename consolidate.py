from tabulate import tabulate
import json 

# Loop through datasets
tasks = ["boolq","cb","copa","rte","wic","wsc","stsb","mrpc"] #+ ["copa","rte","mrpc","stsb"]
seeds = [41,42,43,44,45]
models = ["bert-base-uncased", "google/mobilebert-uncased", "distilbert-base-uncased", "moddistilbert-base-uncased", "albert-base-v2","t5-small","squeezebert/squeezebert-uncased","microsoft/deberta-v3-xsmall" ]
# , "huawei-noah/TinyBERT_General_6L_768D"

task_to_metrics = {
    "boolq" : "accuracy",
    "cb" : "accuracy",
    "wic" : "accuracy",
    "wsc" : "accuracy",
    "copa" : "accuracy",
    "rte" : "accuracy",
    "mrpc" : "accuracy",
    "stsb" : "pearson"
}

global_results = {}

for task in tasks:
    global_results[task] = {}
    for model in models:

        global_results[task][model] = {}
        results_per_seed = []
        for seed in seeds:

            #Load file
            folder = f"./tmp/{task}_{task}_{task}"
            file = f"{folder}/{model.split('/')[-1]}_all_results_{task}_{seed}.json"

            results = json.load(open(file,"r"))
            
            # Get best of 18
            
            best_of_18_score = 0
            best_of_18_full_block = None

            for k,scores in results.items():
                
                if( float( scores[ task_to_metrics[ task ] ] ) > best_of_18_score ):
            
                    best_of_18_score = float( scores[ task_to_metrics[ task ] ] ) 
                    best_of_18_full_block    = scores
                
                
            results_per_seed.append( ( best_of_18_full_block, best_of_18_score ) ) 
            

            #For taking the 3rd epoch's result
            #results_per_seed.append( (  results["2"], results["2"][ task_to_metrics[task] ]  ) )

        #Sort best of 18 scores
        results_per_seed.sort(key=lambda x: x[1])
        assert len(results_per_seed) == 5

        #Pick median
        median_result = results_per_seed[2]
        
        global_results[task][model] = median_result[0]

print(global_results)


# Prepare the table data
table_data = []
for model in models:
    row = [model]
    total_scores = []
    for task in tasks:
        result = global_results[task].get(model, {})
        primary_metric = task_to_metrics[task]
        primary_score = result.get(primary_metric)
        if primary_score is not None:
            # Multiply by 100 and round to 2 decimal places
            primary_score_scaled = round(primary_score * 100, 2)
            total_scores.append(primary_score_scaled)
            additional_metrics = [f"{round(score * 100, 2):.2f}" for key, score in result.items() if key != primary_metric]
            score_text = f"{primary_score_scaled:.2f}"
            if additional_metrics:
                score_text += "/" + "/".join(additional_metrics)
            row.append(score_text)
        else:
            row.append("N/A")
    # Calculate average score for the model, multiplied by 100 and rounded
    if total_scores:
        average_score = round(sum(total_scores) / len(total_scores), 2)
        row.append(f"{average_score:.2f}")
    else:
        row.append("N/A")
    table_data.append(row)

# Define headers
headers = ["Model"] + tasks + ["Average Score"]

# Render the table using tabulate
table = tabulate(table_data, headers=headers, tablefmt="grid")
print(table)

res_file = "./saves/res/speed_and_inference.json"
speed_and_inference_res = json.load( open( res_file, 'r' ) )


# Reference values from bert-base-uncased
bert_params = speed_and_inference_res["bert-base-uncased"]["params"]
bert_time = speed_and_inference_res["bert-base-uncased"]["time"]

# Prepare the data for tabulation
table_data = []
for model, data in speed_and_inference_res.items():
    relative_compression = (1 - (data["params"] / bert_params)) * 100
    inference_speedup = bert_time / data["time"]
    table_data.append([
        model,
        data["time"],
        data["params"],
        f"{inference_speedup:.2f}x",
        f"{relative_compression:.2f}%"
    ])

# Print the table using the tabulate function
print(tabulate(table_data, headers=["Model", "Inference Time (s)", "Parameters", "Inference Speedup", "Compression %"]))
