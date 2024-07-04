import json
import matplotlib.pyplot as plt

# Load JSON data for training losses

mod = "_modular"#"_modular"
with open(f'./saves/res/TRAIN_TRAIN_RES_compressed{mod}_distilbert_cb.json', 'r') as file:
    train_data = json.load(file)
    # Sorting by the numeric part of the key to ensure correct order
    train_losses = [train_data[key]['average_loss'] for key in sorted(train_data, key=lambda x: int(x.split('_')[1]))]

# Load JSON data for evaluation losses
with open(f'./saves/res/EVAL_218_res_compressed{mod}_distilbert_cb.json', 'r') as file:
    eval_data = json.load(file)
    eval_losses = [eval_data[key]['loss'] for key in sorted(eval_data, key=lambda x: int(x))]

num_epochs = len(train_losses)
# Plotting both training and evaluation losses
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', linewidth=1.5)
plt.plot(eval_losses, label='Evaluation Loss', linewidth=1.5)
plt.title('Training vs Evaluation Loss ( Modular Training )')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Reduce the number of x-axis labels if there are many epochs
if num_epochs > 50:
    plt.xticks(range(0, num_epochs, num_epochs // 10))  # Adjust the step as needed
else:
    plt.xticks(range(num_epochs))
plt.savefig(f"compressed{mod}_train_v_eval.png")
