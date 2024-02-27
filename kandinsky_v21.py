import os
import time
import torch
from diffusers import KandinskyPipeline, KandinskyPriorPipeline

DEVICE_CPU = torch.device('cpu:0')
DEVICE_GPU = torch.device('cuda:0')

script_path = os.path.dirname(__file__)
models_path = os.path.join(script_path, 'models')
cache_dir = os.path.join(models_path, 'kandinsky21')


def generate_image_by_prompt(prompt: str, filename: str):
    pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior",
                                                        torch_dtype=torch.float32,
                                                        low_cpu_mem_usage=False,
                                                        cache_dir=cache_dir)
    pipe_prior.to(DEVICE_CPU)
    image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()
    pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1",
                                             torch_dtype=torch.float32,
                                             low_cpu_mem_usage=False,
                                             cache_dir=cache_dir)
    pipe.to(DEVICE_CPU)
    image = pipe(
        prompt,
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        height=512,
        width=512,
        num_inference_steps=100,
    ).images
    image[0].save(filename)


if __name__ == '__main__':
    start_time = time.time()
    print("Started...")
    prompt = "photorealistic a black cat in glasses and boots teaches programming in C++ to a group of hedgehogs"
    filename = "output.png"
    generate_image_by_prompt(prompt, filename)
    print(f"Done. Elapsed time {round((time.time() - start_time), 1)} seconds")
