import argparse
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageOps
import PIL
import numpy as np
import torch

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

class ImageRepainter:
    def __init__(self, device="cuda", model="StableDiffusionInpaintingXL", seed=20):
        self.device = device
        
        if model == "StableDiffusionInpainting":
            model_name = "runwayml/stable-diffusion-inpainting"
        elif model =="StableDiffusionInpaintingXL":
            model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        else:
            raise ValueError("Invalid model")
        
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.generator = torch.Generator(device).manual_seed(seed)

    def repaint_and_save(self, prompt, negative_prompt, init_image, mask_image, strength=1.0, guidance_scale=5.0):
        repainted_image = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=self.generator, strength=strength, guidance_scale=guidance_scale).images[0]
        # To remove any alteration of product from the generation
        unmasked_unchanged_image = self.pipeline.image_processor.apply_overlay(mask_image, init_image, repainted_image)
        return unmasked_unchanged_image

class VideoGenerator:
    def __init__(self, device="cuda", seed=42, prompt = None,model = None):
        self.device = device
        self.generator = torch.Generator(device).manual_seed(seed)
        self.model = model
        if model == "StableVideoDiffusion":
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
            )
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.to(device)
        elif model == "I2VGenXL":
            self.pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
            self.pipeline.to(device)
        else:
            raise ValueError("Invalid model")
        
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
    
    def generate_video_svd(self, image,save_path, decode_chunk_size=8, noise_aug_strength=0.1):
        image = image.resize((1024, 576))
        frames = self.pipeline(image, decode_chunk_size=decode_chunk_size, generator=self.generator, noise_aug_strength=noise_aug_strength).frames[0]
        export_to_video(frames, save_path)
    
    def generate_video_i2v(self, image, prompt, negative_prompt, save_path,num_inference_steps=50, guidance_scale=9.0):
        frames = self.pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=self.generator
        ).frames[0]
        export_to_video(frames, save_path)
    
    def generate_and_save_video(self, image, prompt, negative_prompt,save_path, num_inference_steps=50, guidance_scale=9.0, decode_chunk_size=8, noise_aug_strength=0.1):
        if self.model == "StableVideoDiffusion":
            self.generate_video_svd(image,save_path ,num_inference_steps, guidance_scale, decode_chunk_size, noise_aug_strength)
        elif self.model == "I2VGenXL":
            self.generate_video_i2v(image, prompt, negative_prompt,save_path, num_inference_steps, guidance_scale, decode_chunk_size, noise_aug_strength)
        else:
            raise ValueError("Invalid model")

class ProductShootGenerator:
    def __init__(self, video_generation_model, image_generation_model,image_prompt, image_negative_prompt, video_prompt, video_negative_prompt):
        self.video_generation_model = video_generation_model
        self.image_generation_model = image_generation_model
        self.image_prompt = image_prompt
        self.image_negative_prompt = image_negative_prompt
        self.video_prompt = video_prompt
        self.video_negative_prompt = video_negative_prompt

    def load_mask(self, mask_path, n, x):
        mask_image = load_image(mask_path)
        mask_image = ImageOps.invert(mask_image)
        resized_image = mask_image.resize((mask_image.width // n, mask_image.height // n))
        black_image = Image.new('RGB', (512, 512), (255, 255, 255))
        position = ((black_image.width - resized_image.width) // x, (black_image.height - resized_image.height) // x)
        black_image.paste(resized_image, position)
        mask = black_image
        return mask

    def load_target_image(self, image_path, n, x):
        target_image = load_image(image_path)
        black_image = Image.new('RGB', (512, 512), (0,0, 0))
        target_image = target_image.resize((target_image.width // n, target_image.height // n))
        position = ((black_image.width - target_image.width) // x, (black_image.height - target_image.height) // x)
        black_image.paste(target_image, position)
        target_image = black_image
        return target_image

    def generate_product_image(self, init_image, mask_image,save_path):
        repainter = ImageRepainter()
        generated_product_image = repainter.repaint_and_save(
            prompt = self.image_prompt,
            negative_prompt = self.image_negative_prompt,
            init_image = init_image,
            mask_image = mask_image
        )
        generated_product_image.save(save_path)
        return generated_product_image

    def generate_product_video(self, image,save_path):
        video_generator = VideoGenerator(model=self.video_generation_model)
        video_generator.generate_and_save_video(image, self.video_prompt, self.video_negative_prompt,save_path)
    
def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--init_image_path', type=str, default='/teamspace/studios/this_studio/cropped_results/example1_cropped_no-bg.jpg', help='Path to the initial image')
    parser.add_argument('--mask_image_path', type=str, default='/teamspace/studios/this_studio/cropped_results/example1_cropped_no-bg_mask.jpg', help='Path to the mask image')
    parser.add_argument('--n', type=int, default=2, help='Factor to resize the initial image by')
    parser.add_argument('--x', type=int, default=3, help='Factor to calculate the position to paste the mask')
    parser.add_argument('--video_generation_model', type=str, default='StableVideoDiffusion', help='Model to use for video generation')
    parser.add_argument('--image_generation_model', type=str, default='StableDiffusionInpaintingXL', help='Model to use for image generation')
    parser.add_argument('--image_prompt', type=str, default='product placed in a showroom', help='Prompt for image generation')
    parser.add_argument('--image_negative_prompt', type=str, default="product changed in any form bad anatomy, deformed, ugly, disfigured", help='Negative prompt for image generation')
    parser.add_argument('--video_prompt', type=str, default='zoom in on the scene', help='Prompt for video generation')
    parser.add_argument('--video_negative_prompt', type=str, default="product changed in any form bad anatomy, deformed, ugly, disfigured", help='Negative prompt for video generation')
    parser.add_argument('--generate_image', type=bool, default=True, help='Generate image')
    parser.add_argument('--generate_video', type=bool, default=True, help='Generate video')
    parser.add_argument('--save_path_image', type=str, default="generated_product_image_example3.png", help='Path to save the generated image')
    parser.add_argument('--save_path_video', type=str, default="generated_product_video_example3.mp4", help='Path to save the generated video')
    args = parser.parse_args()
    
    if args.generate_image is True:
        product_shoot_generator = ProductShootGenerator(args.video_generation_model,args.image_generation_model, args.image_prompt, args.image_negative_prompt, args.video_prompt, args.video_negative_prompt)
        product_image = product_shoot_generator.load_target_image(args.init_image_path, args.n, args.x)
        mask_image = product_shoot_generator.load_mask(args.mask_image_path, args.n, args.x)
        if args.save_path_image is not None:
            save_path_image = args.save_path_image
        else:
            save_path_image = "generated_product_image.png"
        generated_product_image = product_shoot_generator.generate_product_image(product_image, mask_image,save_path_image)
        
    if args.generate_video is True:
        if args.save_path_video is not None:
            save_path_video = args.save_path_video
        else:
            save_path_video = "product_video.mp4"
        product_shoot_generator.generate_product_video(generated_product_image, save_path_video)

if __name__ == "__main__":
    main()