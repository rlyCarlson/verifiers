import modal
from modal import method, Secret, Image, App

# Create Modal app
app = App("bfcl-training")

# Define the image with all dependencies
image = (
    Image.debian_slim(python_version="3.11")
    # Install build dependencies first
    .apt_install("git", "build-essential")
    .pip_install("packaging", "wheel", "setuptools")
    # Install core dependencies
    .pip_install(
        [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate>=0.28.0",
            "datasets>=2.0.0",
            "peft>=0.10.0",
            "wandb",
            "openai",
            "uv",
        ]
    )
    # Install vLLM (may include flash-attn internally)
    .pip_install("vllm>=0.4.0")
    # Try flash-attn separately, skip if it fails
    .run_commands(
        "pip install flash-attn>=2.5.0 || echo 'flash-attn install failed, continuing without it'"
    )
    # Install other optional packages
    .pip_install(["deepspeed>=0.14.0", "liger-kernel"])
    # Clone the verifiers repo, install it, and install the tool-test environment
       # Clone the verifiers repo and install it
    .run_commands(
        "pwd" # Install from the verifiers repo
    )
    .run_commands(
        "git clone https://github.com/willccbb/verifiers.git /opt/verifiers && "
        "cd /opt/verifiers && "
        "pip install -e . && "
        "cd / && "
        "vf-install tool-test --from-repo"  # Install from the verifiers repo
    )
    # Set the python path to include the verifiers repo
    .env(
        {
            "PYTHONPATH": "/opt/verifiers",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# Shared volume for model weights and checkpoints
volume = modal.Volume.from_name("bfcl-training-vol", create_if_missing=True)


# Main training class
@app.cls(
    image=image,
    gpu="A100-40GB:4",  # 4 GPUs for training and inference
    volumes={"/data": volume},
    timeout=7200,  # 2 hours timeout
    secrets=[
        Secret.from_name("huggingface"),  # HF_TOKEN
        Secret.from_name("wandb"),  # WANDB_API_KEY
    ],
)
class BFCourse:
    @method()
    def train(
        self,
        model_name: str,
        run_name: str | None,
        max_steps: int = 300,
    ):
        """Train BFCL model using GRPO."""
        import os
        import verifiers as vf

        # Set up environment
        os.environ["OPENAI_API_KEY"] = "dummy"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        os.environ["WANDB_PROJECT"] = "bfcl-rl-training"

        if run_name is None:
            run_name = f"bfcl_modal_{model_name.split('/')[-1]}"

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
        training_args.output_dir = "/data/outputs/" + run_name

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

        # Save final model to volume
        trainer.save_model()

        return training_args.output_dir


@app.local_entrypoint()
def main(
    model_name: str = "Salesforce/xLAM-2-1b-fc-r",
    max_steps: int = 300,
    run_name: str = None,
):
    """Main entry point for BFCL training."""
    print(f"🚀 Starting BFCL training on Modal")
    print(f"📦 Model: {model_name}")
    print(f"🔄 Max steps: {max_steps}")
    print(f"📝 Run name: {run_name or 'auto-generated'}")

    # Run the training
    bf_course = BFCourse()
    result = bf_course.train.remote(
        model_name=model_name,
        run_name=run_name,
        max_steps=max_steps,
    )
    print(f"✅ Training completed: {result}") 