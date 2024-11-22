#!/usr/bin/env python

import os
import random
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = """
# DALL•E 3 XL v2
"""

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max

USE_TORCH_COMPILE = 0
ENABLE_CPU_OFFLOAD = 0


if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "fluently/Fluently-XL-Final",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    
    pipe.load_lora_weights("ehristoforu/dalle-3-xl-v2", weight_name="dalle-3-xl-lora-v2.safetensors", adapter_name="dalle")
    pipe.set_adapters("dalle")

    pipe.to("cuda")
    
    

@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    
    seed = int(randomize_seed_fn(seed, randomize_seed))

    if not use_negative_prompt:
        negative_prompt = ""  # type: ignore

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil",
    ).images
    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

examples = [
    "neon holography crystal cat",
    "a cat eating a piece of cheese",
    "an astronaut riding a horse in space",
    "a cartoon of a boy playing with a tiger",
    "a cute robot artist painting on an easel, concept art",
    "a close up of a woman wearing a transparent, prismatic, elaborate nemeses headdress, over the should pose, brown skin-tone"
]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''
with gr.Blocks(css=css, theme="pseudolab/huggingface-korea-theme") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=False,
    )

    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, preview=True, show_label=False)
    with gr.Accordion("Advanced options", open=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
        negative_prompt = gr.Text(
            label="Negative prompt",
            lines=4,
            max_lines=6,
            value="""(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, (NSFW:1.25)""",
            placeholder="Enter a negative prompt",
            visible=True,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=6,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=False,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )
    

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    
if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_api=False, debug=False)