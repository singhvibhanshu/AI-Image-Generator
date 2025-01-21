import torch
from diffusers import StableDiffusion3Pipeline

# Define the model to be used
model_name = "stabilityai/stable-diffusion-3.5-large"

# Load the pre-trained Stable Diffusion model with optimized settings
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
)

# Enable attention slicing for optimized memory usage
pipeline.enable_attention_slicing()

# Move the pipeline to the GPU for faster processing
pipeline = pipeline.to("cuda")

# Define the text prompt for generating images
text_prompt = "a programmer touching grass"

# Generate images using the pipeline
result = pipeline(
    text_prompt,
    num_inference_steps=20,  # Number of inference steps for image generation
    guidance_scale=3.5,      # Control the adherence to the prompt
    height=512,              # Set the image height
    width=512                # Set the image width
)

# Extract the generated images
generated_images = result.images

# Save each generated image to the local directory
for idx, image in enumerate(generated_images):
    image.save(f"output_image_{idx}.png")  # Save with a clear naming convention
