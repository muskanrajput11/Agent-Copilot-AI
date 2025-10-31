from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

# --- Step 1: Load the Cleaned Dataset ---
print("Loading the cleaned dataset...")
dataset = load_dataset('csv', data_files='training_data.csv', split='train')

# --- Step 2: Load the Pre-trained Model and Tokenizer ---
model_name = "google/flan-t5-small"
print(f"Loading tokenizer and model for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- Step 3: Preprocess and Tokenize the Data ---
def preprocess_function(examples):
    """Prepares the data for the model by tokenizing prompts and completions."""
    inputs = [doc for doc in examples["prompt"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["completion"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing the dataset... This may take a moment.")
# Using a small subset for quick training. 2000 is a good number for this break.
small_dataset = dataset.shuffle(seed=42).select(range(2000))
tokenized_dataset = small_dataset.map(preprocess_function, batched=True)

# --- Step 4: Define Training Arguments and Train the Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

output_dir = "./support_agent_model"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    no_cuda=not torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print(" Starting model training... This will take a significant amount of time. ")
trainer.train()
print("Training complete!")

# --- Step 5: Save the Final Model ---
print(f"Saving the fine-tuned model to '{output_dir}'...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(" Model saved successfully.")
