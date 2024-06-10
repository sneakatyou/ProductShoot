# ProductShoot

ProductShoot is a repository dedicated to generating images and videos for product ad shoots. It uses advanced machine learning models to generate high-quality visual content based on user-defined prompts.

## Mask Generation
Use this notebook to generate image masks:
[Generate Mask](https://colab.research.google.com/github/shreyas-bk/U-2-Net-Demo/blob/master/DEMOS/U_2_Netp_Cropper_Colab.ipynb#scrollTo=CEDlFFvONi5g)

## Usage

To generate images and videos, use the following command:

```sh
python productshoot.py \
    --init_image_path '/teamspace/studios/this_studio/cropped_results/example1_cropped_no-bg.jpg' \
    --mask_image_path '/teamspace/studios/this_studio/cropped_results/example1_cropped_no-bg_mask.jpg' \
    --n 2 \
    --x 3 \
    --video_generation_model 'StableVideoDiffusion' \
    --image_generation_model 'StableDiffusionInpaintingXL' \
    --image_prompt 'product placed in a showroom' \
    --image_negative_prompt 'product changed in any form bad anatomy, deformed, ugly, disfigured' \
    --video_prompt 'zoom in on the scene' \
    --video_negative_prompt 'product changed in any form bad anatomy, deformed, ugly, disfigured' \
    --generate_image True \
    --generate_video True \
    --save_path_image 'generated_product_image_example3.png' \
    --save_path_video 'generated_product_video_example3.mp4'

init_image_path: Path to the initial image.
mask_image_path: Path to the mask image.
n: Factor to resize the initial image by.
x: Factor to calculate the position to paste the mask. Shift Image inside the frame.
video_generation_model: Model to use for video generation. Default is 'StableVideoDiffusion'. StableVideoDiffusion Doesn't support prompting. Use "I2VGenXL" if you intend to condition video based on prompt.
image_generation_model: Model to use for image generation. Default is 'StableDiffusionInpaintingXL'.
image_prompt: Prompt for image generation.
image_negative_prompt: Negative prompt for image generation.
video_prompt: Prompt for video generation.
video_negative_prompt: Negative prompt for video generation.
generate_image: Whether to generate an image. Default is True.
generate_video: Whether to generate a video. Default is True.
save_path_image: Path to save the generated image.
save_path_video: Path to save the generated video.

```