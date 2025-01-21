import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Define the model to be used
model_name = "stabilityai/stable-diffusion-2-1"

# Load the pre-trained Stable Diffusion model with optimized settings
pipeline = StableDiffusionPipeline.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
)

# Update the scheduler for improved image quality and stability
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)

# Enable attention slicing to optimize memory usage
pipeline.enable_attention_slicing()

# Move the pipeline to the GPU for faster processing
pipeline = pipeline.to("cuda")

# Define a list of prompts for generating images
prompts_list = [
    "a programmer touching grass",
    "A dreamlike landscape with floating islands and waterfalls under a starry sky.",
    "A Roman soldier standing guard in front of the Colosseum during sunset.",
    "A cyberpunk character with neon tattoos in a rain-soaked alley."
]

# Generate images using the pipeline
result = pipeline(
    prompts_list,
    num_inference_steps=50,  # Number of inference steps for better detail
    guidance_scale=3.5,      # Control the adherence to the prompts
    height=512,              # Set the image height
    width=512                # Set the image width
)

# Extract the generated images
generated_images = result.images

# Save each generated image to the local directory
for idx, image in enumerate(generated_images):
    image.save(f"output_image_{idx}.png")  # Save images with a clear naming convention
