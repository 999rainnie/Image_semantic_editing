from PIL import Image
from .vis_utils import preprocess_image
import torch
from tqdm import tqdm

@torch.no_grad()
def ddim_next_step(pipe, model_output, timestep, sample):
    timestep, next_timestep = min(timestep - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep] if timestep >= 0 else pipe.scheduler.final_alpha_cumprod
    alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample
    
@torch.no_grad()
def get_ddim_latents(pipe, img_path, prompt, device):
    if img_path is None: return None
    img = Image.open(img_path).resize((512,512)).convert('RGB')
    image = preprocess_image(img).to(device)
    latent = pipe.vae.encode(image.half()).latent_dist.sample() * 0.18215
    
    cond_input = pipe.tokenizer(
        [prompt], 
        padding="max_length",
        max_length = pipe.tokenizer.model_max_length,
        truncation = True,
        return_tensors = "pt",
    )
    cond_embeddings = pipe.text_encoder(cond_input.input_ids.to(device))[0]
    
    all_latent = [latent]
    latent = latent.clone().detach()
    num_ddim_steps = len(pipe.scheduler.timesteps)
    for i in tqdm(range(num_ddim_steps)):
        t = pipe.scheduler.timesteps[num_ddim_steps - i - 1]
        noise_pred = pipe.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
        latent = ddim_next_step(pipe, noise_pred, t, latent)
        all_latent.append(latent)
    return all_latent