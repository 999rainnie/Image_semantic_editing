from diffusers.models.attention_processor import AttnProcessor, Attention
import torch
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch.nn.functional as F 
from .ptp_utils import * 
from torchvision.utils import save_image

MAX_NUM_WORDS = 77

##TODO::change LocalBlend for semantic_edit 
class LocalBlend:
    
    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        # mask = mask[:1] + mask ##NOTE:: what this mean for? original mask blended?
        return mask
    
    def __call__(self, x_t, inv_x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            orig_maps = attention_store["ori"]["down_cross"][2:4] + attention_store["ori"]["up_cross"][:3]
            edit_maps = attention_store["edit"]["down_cross"][2:4] + attention_store["edit"]["up_cross"][:3]
            
            orig_maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in orig_maps]
            orig_maps = torch.cat(orig_maps, dim=1)
            orig_mask = self.get_mask(x_t, orig_maps, self.alpha_layers, True)

            edit_maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in edit_maps]
            edit_maps = torch.cat(edit_maps, dim=1)
            edit_mask = self.get_mask(x_t, edit_maps, self.alpha_layers, True)
            
            ##TODO:: visualize mask
            save_image(orig_mask, 'orig_mask_image.png')
            save_image(edit_mask, 'edit_mask_image.png')
            
            # fuse orig_mask and edit_mask
            mask = torch.logical_and(orig_mask, edit_mask)
            mask = mask[1:].float()
            
            ##TODO::have to get latent blend, -> to the main code..!
            x_t = x_t * mask + inv_x_t * (1 - mask)
        return x_t
       
    def __init__(self, prompts, num_ddim_steps, words, tokenizer, device, substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th
        

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {'ori' : {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}, \
            'edit' : {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}}
    
    def __init__(self, attn_res=[4096, 1024, 256, 64]): 
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

    def __call__(self, attention_map, is_cross, place_in_unet: str, pred_type='ori'): 
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0: # and is_cross
            if attention_map.shape[1] in self.attn_res:
                self.step_store[pred_type][key].append(attention_map)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps(pred_type)
        return attention_map
    
    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def between_steps(self, pred_type='ori'):
        self.attention_store[pred_type] = self.step_store[pred_type]
        self.step_store = self.get_empty_store()

class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def replace_latents(self, x_t, inv_x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, inv_x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str, pred_type: str):
        super(AttentionControlEdit, self)(attn, is_cross, place_in_unet, pred_type)
        if pred_type == 'edit' and (is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1])):
            h = attn.shape[0] // (self.batch_size)
            attn_repalce = attn.reshape(self.batch_size, h, *attn.shape[1:])
            
            # get attn_base
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            attn_base = self.attention_store['orig'][key][-1]
            print("attention from attn_store['orig']: ", attn_base.shape)
            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
            else:
                attn_repalce_new = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn_repalce_new.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]], tokenizer, device,
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        # print('cross replace alpha: ', self.cross_replace_alpha)
        
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer, device,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.mapper = get_replacement_mapper(prompts, tokenizer).to(device)
        
class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer, device,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.mapper, alphas = get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer, device, equalizer, 
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, num_ddim_steps, tokenizer, device, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, num_ddim_steps, blend_words, tokenizer, device)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, tokenizer=tokenizer, device=device, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, tokenizer=tokenizer, device=device, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, tokenizer=tokenizer, device=device, equalizer=eq, local_blend=lb, controller=controller)
    return controller