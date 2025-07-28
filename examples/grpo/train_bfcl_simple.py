import verifiers as vf

"""
Simple BFCL-style training using the tool_test pattern.

This approach:
1. Uses existing tool_test environment pattern
2. Just renames functions to BFCL-style names
3. Focuses on function calling correctness, not execution
4. Much simpler than implementing actual functions

Usage:
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model Salesforce/xLAM-2-1b-fc-r \
    --data-parallel-size 2 --enforce-eager --disable-log-requests \
    --enable-auto-tool-choice

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config-file configs/zero3.yaml \
    --num-processes 2 examples/grpo/train_bfcl_simple.py
"""

# Load existing tool_test environment
vf_env = vf.load_environment("tool-test", num_train_examples=2000, num_eval_examples=200)

# Use xLAM model optimized for function calling  
model_name = "Salesforce/xLAM-2-1b-fc-r"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Training configuration
training_args = vf.grpo_defaults(run_name=f"bfcl_simple_{model_name.split('/')[-1]}")

# Basic settings
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 4
training_args.max_steps = 300
training_args.learning_rate = 5e-7

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)

print("Starting BFCL-style training with existing tool_test environment...")
trainer.train() 