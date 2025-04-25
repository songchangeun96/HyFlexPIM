from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, get_scheduler, LlamaTokenizerFast, DataCollatorWithPadding
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from itertools import chain
import torch
import math
from tqdm import tqdm
import os
from torch import nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import time
import copy

# def tokenize_function(examples):
#     return tokenizer(examples[text_column_name], truncation=True, max_length=512, padding='max_length')
# hyflex_utils.py

def tokenize_function(examples, tokenizer, text_column_name):
    return tokenizer(
        examples[text_column_name],
        truncation=True,
        max_length=512,
        padding="max_length"
    )


    
class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {key: torch.tensor([f[key] for f in features]) for key in features[0]}

        batch["labels"] = torch.full_like(batch["input_ids"], -100)
        batch["labels"][:, :-1] = batch["input_ids"][:, 1:].clone()
        # Ensure padding tokens remain ignored
        batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100

        return batch


#  DiagonalLinear
class DiagonalLinear(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.diag = nn.Parameter(torch.ones(feature_size), requires_grad=True)
    
    def forward(self, input):
        return input @ torch.diag(self.diag) 
    
    def set_value(self, value):
        length = value.shape[0]
        self.diag.data[:length] = value
        self.diag.data.requires_grad_(True) 

#  MaskedLinear
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)
    
    def forward(self, input):
        weight = self.weight * self.mask
        return nn.functional.linear(input, weight, self.bias)
    
    def set_mask(self, mask):
        height, width = mask.shape
        self.mask.data[:height, :width] = mask
        self.mask.data.requires_grad_(False)
        
    def set_value(self, weight):
        height, width = weight.shape
        self.weight.data[:height, :width] = weight
        self.weight.data.requires_grad_(True)

#  DecomposeLinear
class DecomposeLinear(nn.Module):
    def __init__(self, linear_layer, rank=None):
        super().__init__()
        
        cur_device = linear_layer.weight.device
        
        linear_weight = linear_layer.weight.detach()

        if rank is None:
            self.rank = min(linear_weight.shape)
        else:
            self.rank = rank

        self.full_rank_dims = min(linear_weight.shape)
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.device = linear_layer.weight.device
    
        # Init masked linear layer
        self.U = MaskedLinear(self.rank, linear_layer.out_features, bias=linear_layer.bias is not None).to(cur_device)
        self.Sigma = DiagonalLinear(self.rank).to(cur_device)
        self.V = MaskedLinear(linear_layer.in_features, self.rank, bias=False).to(cur_device)
        
        # Use SVD to compute the basis and its coordinates
        U, Sigma, V = self.SVD(linear_weight)

        # Assign value to U matrix
        self.U.set_value(U[:, :self.rank].to(cur_device))

        #  weight   (    )
        self.weight = linear_layer.weight  #  weight 

        #  bias   
        if linear_layer.bias is not None:
            self.U.bias.data = linear_layer.bias.detach().to(cur_device)

        # Assign value to Sigma Diagonal matrix
        self.Sigma.set_value(Sigma[:self.rank].to(cur_device))

        # Assign value to V matrix
        self.V.set_value(V[:self.rank, :].to(cur_device))
        
        # Save product of V and Sigma
        self.vs = torch.tensor(0.0).to(cur_device)
            
    def forward(self, input):
        self.vs = self.V.weight.T @ torch.diag(self.Sigma.diag)
        h = input @ self.vs
        h = self.U(h)
        return h

    def SVD(self, mats):
        return torch.linalg.svd(mats, full_matrices=False)


#  LLaMA  Linear → DecomposeLinear  
def replace_linear_layer_llama(model, rank=None):
    """
    Recursively replaces Linear layers with DecomposeLinear in LlamaForCausalLM.
    """
    
    def replace_linear_layer_helper(module, rank=None):
        for name, child in module.named_children():
            module_name = type(module).__name__

            if isinstance(child, nn.Linear) and module_name != "DecomposeLinear":
                # Automatically determine rank if not specified
                if rank is None:
                    r, c = tuple(child.weight.size())
                    rank = r * c // (r + c)  # Heuristic rank calculation
                
                print(f"Replacing {module_name}.{name} with DecomposeLinear, rank={rank}")
                l = module.__getattr__(name)
                module.__setattr__(name, DecomposeLinear(l, rank))

                #  weight   (weight    )
                assert hasattr(module.__getattr__(name), "weight"), f"{name} does not have a weight attribute!"
            
            else:
                replace_linear_layer_helper(child, rank)

    #  LLaMA  `Linear`    
    for layer in model.model.layers:  # LlamaModel  Transformer 
        replace_linear_layer_helper(layer, rank)

    #  lm_head 
    if isinstance(model.lm_head, nn.Linear):
        print(f"Replacing lm_head with DecomposeLinear, rank={rank}")
        model.lm_head = DecomposeLinear(model.lm_head, rank)

    print(" All Linear layers replaced successfully!")




#  Gradient  
def save_gradient(layer, layer_name, i):
    if hasattr(layer, 'Sigma') and layer.Sigma.diag.grad is not None:
        save_path = f'./gradient/pre_final_batch_epoch1_layer{i}_{layer_name}_Sigma_diag_grad.pth'
        os.makedirs("./gradient", exist_ok=True)
        torch.save(layer.Sigma.diag.grad, save_path)
        print(f" Saved gradient for layer {i} {layer_name} Sigma.")
        
    
#  Inference     
def evaluate_model(model, tokenizer, eval_dataloader):
    import torch.nn.functional as F
    import math
    from tqdm import tqdm

    total_eval_loss = 0
    total_tokens = 0
    model.to(torch.float32)
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum"
            )

            total_eval_loss += loss.item()
            total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()

    avg_eval_loss = total_eval_loss / total_tokens
    eval_perplexity = math.exp(avg_eval_loss)

    print(f"Validation Loss: {avg_eval_loss:.4f}, Validation Perplexity: {eval_perplexity:.4f}")
    return avg_eval_loss, eval_perplexity
    
#  Training w/ gradient saving
def train_model_gradient_saving(model, tokenizer, optimizer, train_dataloader, eval_dataloader, accelerator, epochs=3, grad_accum_steps=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

    model.train()

    for epoch in range(epochs):
        total_train_loss = 0
        total_tokens = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum"
            )

            non_padding_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            loss = loss / non_padding_tokens

            accelerator.backward(loss)

            if (step + 1) % grad_accum_steps == 0:
                is_last_step = (step + 1 == len(train_dataloader))

                if is_last_step:
                    for i, layer in enumerate(model.model.layers):
                        save_gradient(layer.self_attn.q_proj, "query", i)
                        save_gradient(layer.self_attn.k_proj, "key", i)
                        save_gradient(layer.self_attn.v_proj, "value", i)
                        save_gradient(layer.self_attn.o_proj, "projection", i)
                        save_gradient(layer.mlp.gate_proj, "fc1", i)
                        save_gradient(layer.mlp.down_proj, "fc2", i)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            total_train_loss += loss.item()
            total_tokens += non_padding_tokens

            if step % 5000 == 0 and step != 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                total_eval_loss = 0
                total_eval_tokens = 0
                model.eval()

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = batch["input_ids"][:, 1:].contiguous()

                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=tokenizer.pad_token_id,
                            reduction="sum"
                        )

                        total_eval_loss += loss.item()
                        total_eval_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()

                avg_eval_loss = total_eval_loss / total_eval_tokens
                eval_perplexity = math.exp(avg_eval_loss)
                print(f"Validation Loss: {avg_eval_loss:.4f}, Validation Perplexity: {eval_perplexity:.4f}")
                model.train()

        avg_train_loss = total_train_loss / total_tokens
        train_perplexity = math.exp(avg_train_loss)
        print(f"Epoch {epoch+1} Completed. Avg Train Loss: {avg_train_loss:.4f}, Perplexity: {train_perplexity:.4f}")

    print("Training Complete!")


#  Training
def train_model(model, tokenizer, optimizer, train_dataloader, eval_dataloader, accelerator, epochs=3, grad_accum_steps=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        total_tokens = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")

        for step, batch in enumerate(progress_bar):
            batch = {k: v for k, v in batch.items()}

            #  Forward pass
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            #  Shift logits and labels for causal language modeling
            shift_logits = logits[:, :-1, :].contiguous()  # Remove last token's logits
            shift_labels = batch["input_ids"][:, 1:].contiguous()  # Remove first token's labels

            #  Compute loss manually
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # Flatten logits
                shift_labels.view(-1),  # Flatten labels
                ignore_index=tokenizer.pad_token_id,  # Ignore padding
                reduction="sum"  # Sum instead of mean
            )

            #  Normalize loss by the number of tokens
            non_padding_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            loss = loss / non_padding_tokens 

            #  Backpropagation
            accelerator.backward(loss)

            #  Gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


            #  Track total loss
            total_train_loss += loss.item()
            total_tokens += non_padding_tokens

            #  Print progress
            if step % 5000 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                total_eval_loss = 0
                total_tokens = 0

                model.eval()
                for batch in tqdm(eval_dataloader):
                    batch = {k: v for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                        # Shift logits and labels to align with causal LM objective
                        shift_logits = logits[:, :-1, :].contiguous()  # Remove last token's logits
                        shift_labels = batch["input_ids"][:, 1:].contiguous()  # Remove first token's labels
                        
                        # Compute loss manually
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),  # Flatten logits
                            shift_labels.view(-1),  # Flatten labels
                            ignore_index=tokenizer.pad_token_id,  # Ignore padding
                            reduction="sum"  # Sum instead of mean
                        )

                        # Accumulate loss and count valid tokens
                        total_eval_loss += loss.item()
                        total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()  # Count non-padding tokens

                # Compute average loss per token
                avg_eval_loss = total_eval_loss / total_tokens
                eval_perplexity = math.exp(avg_eval_loss)
        
                print(f"Validation Loss: {avg_eval_loss:.4f}, Validation Perplexity: {eval_perplexity:.4f}")
                model.train()

        #  Compute Perplexity after each epoch
        avg_train_loss = total_train_loss / total_tokens
        train_perplexity = math.exp(avg_train_loss)

        print(f"Epoch {epoch+1} Completed. Avg Train Loss: {avg_train_loss:.4f}, Perplexity: {train_perplexity:.4f}")

    print("Training Complete!")
        


torch.manual_seed(int(time.time())) 

#  Gradient  Noise Injection 
def adjust_std(eigenvalues, original_std, threshold1, threshold2):
    adjusted_std_case1 = original_std
    adjusted_std_case2 = 0
    adjusted_std_case3 = 0

    adjusted_std = torch.where(eigenvalues <= threshold1, adjusted_std_case1, original_std)
    adjusted_std = torch.where((eigenvalues > threshold1) & (eigenvalues <= threshold2), adjusted_std_case2, adjusted_std)
    adjusted_std = torch.where(eigenvalues > threshold2, adjusted_std_case3, adjusted_std)

    return adjusted_std


#  Clipping Value  (Threshold )
def get_clipping_value(tensor):
    flat_tensor = tensor.flatten()
    sorted_tensor = torch.sort(flat_tensor, descending=True)[0]
    return sorted_tensor[1]  # Change 1 to others for better quantization range     


#  MSB, MID2, MID, LSB  Quantization 
def uniform_quantization(tensor, clip_val, bit):
    scale = (2 ** (bit - 1)) - 1
    tensor_q = tensor.clamp(-clip_val, clip_val) / clip_val * scale
    tensor_q = (tensor_q.round() - tensor_q).detach() + tensor_q  # STE 

    tensor_q_int = tensor_q.to(torch.int8)
    msb_2_bits = (tensor_q_int & 0xC0)
    mid_2_bits = (tensor_q_int & 0x30)
    mid2_2_bits = (tensor_q_int & 0x0C)
    lsb_4_bits = (tensor_q_int & 0x03)

    msb_2_bits_scaled = msb_2_bits / scale * clip_val
    mid_2_bits_scaled = mid_2_bits / scale * clip_val
    mid2_2_bits_scaled = mid2_2_bits / scale * clip_val
    lsb_4_bits_scaled = lsb_4_bits / scale * clip_val

    return msb_2_bits_scaled, mid_2_bits_scaled, mid2_2_bits_scaled, lsb_4_bits_scaled

#   Tensor Noise  (Gradient )
def noise_inject_tensor(weight_tensor, mask, std, device):
    std = torch.tensor(std, device=device)  #  float → torch.tensor 

    #  mask  weight_tensor  (row, col   )
    if mask.dim() == 1:  
        if weight_tensor.shape[0] == mask.shape[0]:  # Mask  (Row)  
            mask = mask.unsqueeze(1)  # (N,) → (N, 1)
        elif weight_tensor.shape[1] == mask.shape[0]:  # Mask  (Col)  
            mask = mask.unsqueeze(0)  # (N,) → (1, N)

    #  Noise 
    noise = 1. + std * torch.randn_like(weight_tensor)
    injected_weight = weight_tensor * (mask + (1 - mask) * noise)

    return injected_weight.to(device)



#  Gradient  Noise Injection  (Threshold )
def adjust_std_with_threshold(gradients, std, th):
    sorted_grads, indices = torch.sort(gradients, descending=True)  # Gradient   
    th_index = int(len(sorted_grads) * th / 100)  #  th%   

    mask = torch.zeros_like(gradients)  #  0 (Noise )
    mask[indices[:th_index]] = 1  #  `th%` 1  (Noise=0 )

    return mask  # Mask  (1: Noise=0, 0: Noise )



def compute_mape(original, noisy):
    mask = original != 0  # 0   
    if mask.sum() == 0:  #    0,  0% 
        return 0.0
    
    return (torch.abs(original[mask] - noisy[mask]) / (torch.abs(original[mask]) + 1e-8)).mean().item() * 100


def apply_noise_to_llama(model, std, th, device="cuda"):
    num_layers = len(model.model.layers)  # LLaMA Transformer Block 

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        for name, sub_layer in [
            ("query", attn.q_proj), ("key", attn.k_proj), ("value", attn.v_proj), 
            ("projection", attn.o_proj), ("fc1", mlp.gate_proj), ("fc2", mlp.down_proj)
        ]:
            if isinstance(sub_layer, DecomposeLinear):
                #  Gradient  (Sigma Diagonal )
                gradients = sub_layer.Sigma.diag.grad

                #  `grad` None 0 
                if gradients is None:
                    print(f"🚨 Warning: Gradients are None for {name} in layer {i}. Using zeros instead.")
                    gradients = torch.zeros_like(sub_layer.Sigma.diag, device=device)

                gradients = gradients.abs()
                mask = adjust_std_with_threshold(gradients, std, th)

                #  Clipping  
                clip_val_U = get_clipping_value(sub_layer.U.weight)
                clip_val_VS = get_clipping_value(torch.diag(sub_layer.Sigma.diag) @ sub_layer.V.weight)

                #  Noise     
                original_MSB_U, original_MID_U, original_MID2_U, original_LSB_U = uniform_quantization(sub_layer.U.weight, clip_val_U, 8)
                original_MSB_VS, original_MID_VS, original_MID2_VS, original_LSB_VS = uniform_quantization(
                    (torch.diag(sub_layer.Sigma.diag) @ sub_layer.V.weight), clip_val_VS, 8
                )

                #  Quantization  (MSB, MID2, MID, LSB )
                temp_MSB_U, temp_MID_U, temp_MID2_U, temp_LSB_U = uniform_quantization(sub_layer.U.weight, clip_val_U, 8)
                temp_MSB_VS, temp_MID_VS, temp_MID2_VS, temp_LSB_VS = uniform_quantization(
                    (torch.diag(sub_layer.Sigma.diag) @ sub_layer.V.weight), clip_val_VS, 8
                )

                #  Noise Injection  (MSB, MID2, MID, LSB   std )
                temp_MSB_U = noise_inject_tensor(temp_MSB_U, mask, std, device)
                temp_MID_U = noise_inject_tensor(temp_MID_U, mask, std, device)
                temp_MID2_U = noise_inject_tensor(temp_MID2_U, mask, std, device)
                temp_LSB_U = noise_inject_tensor(temp_LSB_U, mask, std, device)

                temp_MSB_VS = noise_inject_tensor(temp_MSB_VS, mask, std, device)
                temp_MID_VS = noise_inject_tensor(temp_MID_VS, mask, std, device)
                temp_MID2_VS = noise_inject_tensor(temp_MID2_VS, mask, std, device)
                temp_LSB_VS = noise_inject_tensor(temp_LSB_VS, mask, std, device)

                #  MAPE (Mean Absolute Percentage Error) 
                mape_MSB_U = compute_mape(original_MSB_U, temp_MSB_U)
                mape_MID_U = compute_mape(original_MID_U, temp_MID_U)
                mape_MID2_U = compute_mape(original_MID2_U, temp_MID2_U)
                mape_LSB_U = compute_mape(original_LSB_U, temp_LSB_U)

                mape_MSB_VS = compute_mape(original_MSB_VS, temp_MSB_VS)
                mape_MID_VS = compute_mape(original_MID_VS, temp_MID_VS)
                mape_MID2_VS = compute_mape(original_MID2_VS, temp_MID2_VS)
                mape_LSB_VS = compute_mape(original_LSB_VS, temp_LSB_VS)

                #      
                sub_layer.U.weight.data = (temp_MSB_U + temp_MID_U + temp_MID2_U + temp_LSB_U).to(device)
                sub_layer.Sigma.diag.data = torch.diag(torch.eye(temp_MSB_U.shape[1])).to(device)
                sub_layer.V.weight.data = (temp_MSB_VS + temp_MID_VS + temp_MID2_VS + temp_LSB_VS).to(device)

                #    MAPE 
                print(f" Layer {i}, {name}:")
                print(f"   - MSB_U  Noise Impact: {mape_MSB_U:.2f}%")
                print(f"   - MID_U  Noise Impact: {mape_MID_U:.2f}%")
                print(f"   - MID2_U Noise Impact: {mape_MID2_U:.2f}%")
                print(f"   - LSB_U  Noise Impact: {mape_LSB_U:.2f}%")
                print(f"   - MSB_VS Noise Impact: {mape_MSB_VS:.2f}%")
                print(f"   - MID_VS Noise Impact: {mape_MID_VS:.2f}%")
                print(f"   - MID2_VS Noise Impact: {mape_MID2_VS:.2f}%")
                print(f"   - LSB_VS Noise Impact: {mape_LSB_VS:.2f}%")
                print("-" * 50)

    return model.to(device)

def evaluate_llama(model, eval_dataloader, tokenizer):
    total_eval_loss = 0
    total_tokens = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.float32).to(device)
    model.eval()

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum"
            )

            total_eval_loss += loss.item()
            total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()

    avg_eval_loss = total_eval_loss / total_tokens
    eval_perplexity = math.exp(avg_eval_loss)

    print(f"Validation Loss: {avg_eval_loss:.4f}, Validation Perplexity: {eval_perplexity:.4f}")
    return avg_eval_loss, eval_perplexity



def load_gradients(model):
    """  gradient LLaMA    """
    for i, layer in enumerate(model.model.layers):
        try:
            layer.self_attn.q_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_query_Sigma_diag_grad.pth')
            layer.self_attn.k_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_key_Sigma_diag_grad.pth')
            layer.self_attn.v_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_value_Sigma_diag_grad.pth')
            layer.self_attn.o_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_projection_Sigma_diag_grad.pth')
            layer.mlp.gate_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_fc1_Sigma_diag_grad.pth')
            layer.mlp.down_proj.Sigma.diag.grad = torch.load(f'./gradient/pre_final_batch_epoch1_layer{i}_fc2_Sigma_diag_grad.pth')

        except FileNotFoundError as e:
            print(f"🚨 Warning: Missing gradient file for layer {i}. Error: {e}")

    print(" All Gradients Loaded Successfully!")

