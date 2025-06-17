from conch.open_clip_custom import create_model_from_pretrained
import torch
import torch.nn as nn
import timm
import os
from torchvision import transforms
# from huggingface_hub import login, hf_hub_download

# class CONCHModelForFinetuning(nn.Module):
#     def __init__(self, num_classes=2, hidden_size=512, checkpoint_path=None):
#         super().__init__()
#         self.model = self.make_conch()
#         self.fc = nn.Linear(hidden_size, num_classes)
#         if checkpoint_path:
#             self.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))

#     def make_conch(self):
#         model, _ = create_model_from_pretrained(
#             'conch_ViT-B-16',
#             r"C:\Users\Vivian\Documents\CONCH\checkpoints\conch\pytorch_model.bin",
#             device=torch.device('cuda')
#         )
#         return model

#     def forward(self, x):
#         out, _ = self.model.visual(x)
#         return self.fc(out)


class CONCHModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2, config={'hidden_size': 512}): # change number of classes for each dataset(8 for breast, 2 for breast)
        super().__init__()
        self.config = config
        self.model = self.make_conch()
        # self.fc = nn.Linear(self.config['hidden_size'], num_classes) # full finetuning?

        # linear probing
        # Freeze all parameters in the backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Only this linear layer will be trained
        self.fc = nn.Linear(self.config['hidden_size'], num_classes)

    def make_conch(self):
        # Load the model from "create_model_from_pretrained"
        model_cfg = 'conch_ViT-B-16'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # checkpoint_path = 'checkpoints/CONCH/pytorch_model.bin'
        checkpoint_path = 'C:\\Users\\Vivian\\Documents\\CONCH\\checkpoints\\conch\\pytorch_model.bin' # load in checkpoint here
        # checkpoint_path = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights\Fold2_F_PT_model.pth' # loading breakhis finetuned model
        model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
        
        return model
        
    def forward(self, x):
        out, h = self.model.visual(x)
        return self.fc(out)



class UNIModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2,checkpoint_path=None): # change number of classes accordingly 
        super().__init__()
        # self.config = config
        self.model = self.make_uni()
        # self.fc = nn.Linear(self.config['hidden_size'], num_classes) # keep commented
        # self.fc = nn.Linear(1024, num_classes)  # Match Vision Transformer output # full finetuning

        #***** Freeze all backbone parameters - linear probing *****
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a trainable classification head
        self.fc = nn.Linear(1024, num_classes)

        # ----- Load checkpoint if needed -----
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    
    def make_uni(self):
        local_dir = r"C:\Users\Vivian\Documents\CONCH\checkpoints\uni" 
        os.makedirs(local_dir, exist_ok=True)  
        
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        
        return model 
    
    def forward(self, x):
        out = self.model(x)
        return self.fc(out)



class UNI2ModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2): # change number of classes for each dataset
        super().__init__()
        # self.config = config
        self.model = self.make_uni2()
        # self.fc = nn.Linear(1536, num_classes)  # full finetuning

        # Freeze all backbone parameters for linear probing
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(1536, num_classes)  

    def make_uni2(self):
        local_dir = 'C:\\Users\\Vivian\\Documents\\UNI2\\UNI\\assets\\ckpts\\uni2-h' 
        os.makedirs(local_dir, exist_ok=True) 
       
        timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
        }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        return model 
        
    def forward(self, x):
        out = self.model(x)
        return self.fc(out)