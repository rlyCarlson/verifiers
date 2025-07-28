import verifiers as vf

"""
Berkeley Function Calling Leaderboard (BFCL) training script for Salesforce/xLAM-2-1b-fc-r

This script trains a model on function calling tasks similar to those in BFCL,
using GRPO (Group Relative Policy Optimization) to improve tool usage.

Setup:
1. Install environment: 
   vf-install bfcl-env

2. Start vLLM inference server (separate terminal):
   CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model Salesforce/xLAM-2-1b-fc-r \
       --data-parallel-size 2 --enforce-eager --disable-log-requests \
       --enable-auto-tool-choice

3. Run training:
   CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 \
       --config-file configs/zero3.yaml examples/grpo/train_bfcl.py

Key features of this training:
- Uses the xLAM model which is pre-trained for function calling
- BFCL-style mathematical and utility functions
- GRPO training to improve function call quality
- Configurable for different model sizes and batch configurations
"""

def main():
    # Load the BFCL environment
    print("Loading BFCL environment...")
    vf_env = vf.load_environment(
        env_id="bfcl-env", 
        num_train_examples=2000,  # Increase for longer training
        num_eval_examples=200
    )
    
    # Model configuration - using xLAM which is optimized for function calling
    model_name = "Salesforce/xLAM-2-1b-fc-r"
    run_name = f"bfcl_{model_name.split('/')[-1].lower()}"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    
    # GRPO training configuration
    training_args = vf.grpo_defaults(run_name=run_name)
    
    # Batch configuration - adjust based on your GPU memory
    training_args.per_device_train_batch_size = 8    # Prompts per GPU
    training_args.num_generations = 8                # Completions per prompt (group size)
    training_args.gradient_accumulation_steps = 4    # Steps before optimizer update
    
    # Model and generation settings
    training_args.max_tokens = 1024                  # Max tokens per generation
    training_args.max_seq_len = 2048                 # Max sequence length
    training_args.temperature = 0.8                  # Sampling temperature
    training_args.top_p = 0.9                       # Nucleus sampling
    
    # Training schedule
    training_args.max_steps = 500                    # Total training steps
    training_args.learning_rate = 5e-7               # Lower LR for fine-tuning
    training_args.warmup_steps = 20                  # Warmup steps
    training_args.weight_decay = 0.01                # Weight decay
    
    # Evaluation settings
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 50
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    
    # GRPO-specific settings
    training_args.epsilon = 0.2                      # PPO clipping parameter
    training_args.beta = 0.01                        # KL penalty coefficient
    training_args.loss_type = "dr_grpo"              # Dr. GRPO loss (recommended)
    training_args.scale_rewards = False              # Don't scale rewards (per Dr. GRPO paper)
    
    # Logging settings
    training_args.log_completions = True             # Log sample completions
    training_args.num_completions_to_print = 5       # Number to print
    training_args.logging_steps = 10                 # Log every N steps
    
    # Environment-specific settings
    training_args.mask_env_responses = True          # Mask environment responses from loss
    training_args.mask_truncated_completions = True  # Mask truncated completions
    
    # Async generation settings (for better GPU utilization)
    training_args.num_batches_ahead = 1              # Generate batches ahead of time
    training_args.max_concurrent = 512               # Max concurrent environment calls
    
    print("Training configuration:")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.num_generations * training_args.gradient_accumulation_steps * 2}")  # * num_processes
    print(f"  Max steps: {training_args.max_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Environment: {len(vf_env.dataset)} training examples")
    
    # Create trainer with LoRA for efficient fine-tuning
    print("Creating GRPO trainer...")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=vf.lora_defaults(),  # Use LoRA for efficient training
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    
    # Optional: Push to Hub
    if hasattr(training_args, 'push_to_hub') and training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main() 