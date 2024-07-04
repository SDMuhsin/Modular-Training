
import json 

# Loop through datasets
tasks = ["boolq","cb","wic","wsc"] #+ ["copa","rte","mrpc","stsb"]
seeds = [41,42,43,44,45]
models = ["bert-base-uncased", "huawei-noah/TinyBERT_General_6L_768D", "google/mobilebert-uncased", "distilbert-base-uncased", "moddistilbert-base-uncased"]


task_to_metrics = {
    "boolq" : "accuracy",
    "cb" : "accuracy",
    "wic" : "accuracy",
    "wsc" : "accuracy"
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

        #Sort best of 18 scores
        results_per_seed.sort(key=lambda x: x[1])
        assert len(results_per_seed) == 5

        #Pick median
        median_result = results_per_seed[2]
        
        global_results[task][model] = median_result[0]

print(global_results)

def print_table(global_results):
    # Find all unique model names
    models = set()
    for dataset in global_results.values():
        for model in dataset:
            models.add(model)

    models = sorted(models)  # Sort model names
    headers = ["Dataset"] + models

    # Determine column widths by finding the maximum length of contents for each column
    column_widths = [len(header) for header in headers]
    for dataset, results in global_results.items():
        column_widths[0] = max(column_widths[0], len(dataset))
        for idx, model in enumerate(models, start=1):
            if model in results:
                metrics = results[model]
                metrics_str = " / ".join(f"{key}: {metrics[key]:.3f}" for key in sorted(metrics.keys()))
            else:
                metrics_str = "N/A"
            column_widths[idx] = max(column_widths[idx], len(metrics_str))

    # Create the format string for each row based on column widths
    row_format = "| " + " | ".join(f"{{:<{width}}}" for width in column_widths) + " |"

    # Print the table with proper alignment
    print("-" * (sum(column_widths) + 3 * len(column_widths) + 1))
    print(row_format.format(*headers))
    print("-" * (sum(column_widths) + 3 * len(column_widths) + 1))

    for dataset, results in global_results.items():
        row = [dataset]
        for model in models:
            if model in results:
                metrics = results[model]
                metrics_str = " / ".join(f"{key}: {metrics[key]:.4f}" for key in sorted(metrics.keys()))
            else:
                metrics_str = "N/A"
            row.append(metrics_str)
        print(row_format.format(*row))
        print("-" * (sum(column_widths) + 3 * len(column_widths) + 1))

print_table(global_results)
