from conch.open_clip_custom import create_model_from_pretrained

# model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "C:\\Users\\Vivian\\Documents\\CONCH\\pytorch_model.bin")
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")

