
"""
This script is a Colab-friendly version for training a BFCL model using GRPO.
It is adapted from modal_bfcl_training.py.

## Colab Setup

Before running this script, you need to set up your Colab environment.

1.  **Set up the GPU:**
    Go to "Runtime" -> "Change runtime type" and select a GPU accelerator (e.g., T4, A100).

2.  **Set up secrets:**
    You'll need a Hugging Face token to download models and a Weights & Biases API key for logging.
    In Colab, you can use the secrets manager (key icon on the left panel).
    - Create a secret named 'HF_TOKEN' with your Hugging Face token.
    - Create a secret named 'WANDB_API_KEY' with your W&B API key.

    Then, you can load them into your environment like this:
    ```python
    from google.colab import userdata
    import os

    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
    ```

3.  **Install dependencies:**
    Run the following commands in a Colab cell:
    ```bash
    !pip install torch>=2.0.0 transformers>=4.40.0 accelerate>=0.28.0 datasets>=2.0.0 peft>=0.10.0 wandb openai uv deepspeed>=0.14.0 liger-kernel vllm>=0.4.0
    !pip install flash-attn>=2.5.0 --no-build-isolation || echo 'flash-attn install failed, continuing without it'

    !git clone https://github.com/willccbb/verifiers.git
    !cd verifiers && pip install -e .
    !vf-install tool-test --from-repo
    ```
4. **Set Python Path**
    To ensure the verifiers library is found, add it to the python path.
    ```python
    import sys
    sys.path.append('/content/verifiers')
    ```
"""

import os
import argparse
import verifiers as vf

def train_bfcl(
    model_name: str,
    run_name: str | None,
    max_steps: int = 300,
):
    """Train BFCL model using GRPO."""

    # Set up environment
    os.environ["OPENAI_API_KEY"] = "dummy"
    # In Colab, typically you have one GPU, device 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_PROJECT"] = "bfcl-rl-training"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set HF token from environment if available
    if 'HF_TOKEN' in os.environ:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_TOKEN']


    if run_name is None:
        run_name = f"bfcl_colab_{model_name.split('/')[-1]}"

    print(f"Loading model: {model_name}")
    print(f"Run name: {run_name}")

    # Load environment
    print("Loading tool-test environment...")
    vf_env = vf.load_environment(
        "tool-test", num_train_examples=2000, num_eval_examples=200
    )

    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Configure training
    training_args = vf.grpo_defaults(run_name=run_name)

    # Training hyperparameters
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 8
    training_args.gradient_accumulation_steps = 8
    training_args.max_steps = max_steps
    training_args.learning_rate = 5e-7
    training_args.warmup_steps = 10

    # Evaluation settings
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 50
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.output_dir = "./outputs/" + run_name

    # Generation settings
    training_args.max_tokens = 1024
    training_args.max_seq_len = 2048
    training_args.temperature = 0.8

    # GRPO settings
    training_args.epsilon = 0.2
    training_args.beta = 0.01
    training_args.loss_type = "dr_grpo"
    training_args.scale_rewards = False

    # Async generation
    training_args.num_batches_ahead = 1
    training_args.max_concurrent = 256

    # Logging
    training_args.log_completions = True
    training_args.logging_steps = 10
    training_args.report_to = "wandb"

    print("Creating GRPO trainer...")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=vf.lora_defaults(),  # Use LoRA for efficiency
    )

    print("Starting training...")
    trainer.train()

    print(f"Training completed! Model saved to: {training_args.output_dir}")

    # Save final model
    trainer.save_model()

    return training_args.output_dir

def main():
    parser = argparse.ArgumentParser(description="Train a BFCL model on Colab.")
    parser.add_argument("--model_name", type=str, default="Salesforce/xLAM-2-1b-fc-r", help="The model to train.")
    parser.add_argument("--max_steps", type=int, default=300, help="Number of training steps.")
    parser.add_argument("--run_name", type=str, default=None, help="A name for the training run.")

    args = parser.parse_args()

    # Check for secrets and provide instructions if not found
    if 'HF_TOKEN' not in os.environ or 'WANDB_API_KEY' not in os.environ:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! HF_TOKEN or WANDB_API_KEY not found in environment.  !!!")
        print("!!! Make sure to set them as secrets in Colab.         !!!")
        print("!!! See instructions in the script's docstring.        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Exiting because secrets are required for this script to run properly
        return

    print(f"🚀 Starting BFCL training")
    print(f"📦 Model: {args.model_name}")
    print(f"🔄 Max steps: {args.max_steps}")
    print(f"📝 Run name: {args.run_name or 'auto-generated'}")

    output_dir = train_bfcl(
        model_name=args.model_name,
        run_name=args.run_name,
        max_steps=args.max_steps,
    )

    print(f"✅ Training completed. Final model saved in {output_dir}")


if __name__ == "__main__":
    main() 