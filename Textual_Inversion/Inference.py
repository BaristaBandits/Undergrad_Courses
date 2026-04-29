import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion XL with Textual Inversion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to textual inversion checkpoint")
    parser.add_argument("--token", type=str, required=True, help="Placeholder token to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    args = parser.parse_args()

    # Load SDXL pipeline with safety checker disabled
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    ).to("cuda")

    # Load learned embeddings for both base and refiner tokenizers
    pipeline.load_textual_inversion(
        args.checkpoint,
        token=args.token,
        tokenizer=pipeline.tokenizer  # base text encoder tokenizer
    )
    pipeline.load_textual_inversion(
        args.checkpoint,
        token=args.token,
        tokenizer=pipeline.tokenizer_refiner  # refiner text encoder tokenizer
    )

    # Prompt setup
    bajirao_templates = [
        f"a cinematic photo of {args.token} from Bajirao Mastani",
        f"a dramatic photo of {args.token} wearing Maratha armor",
        f"a movie still of {args.token} in a battlefield scene",
        f"a historical portrait of {args.token} with royal attire",
        f"a close-up of {args.token} with a fierce expression",
        f"a dynamic shot of {args.token} as Bajirao",
        f"a royal photo of {args.token} with a sword",
    ]

    subjects = ["bajirao1","bajirao2", "bajirao3", "bajirao4", "bajirao5", "bajirao6", "bajirao7"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject, prompt in zip(subjects, bajirao_templates):
        image = pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        image.save(output_dir / f"{subject}.png")
        print(f"Saved: {subject}.png | Prompt: {prompt}")

if __name__ == "__main__":
    main()

