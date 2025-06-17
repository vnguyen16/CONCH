class Config:
    num_classes = 2
    hidden_size = 512
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 5
    patience = 5
    device = 'cuda'
    
    # Paths
    base_checkpoint_path = r"C:\Users\Vivian\Documents\CONCH\checkpoints\conch\pytorch_model.bin"
    finetuned_conch_checkpoint = r"C:\Users\Vivian\Documents\CONCH\_finetune_weights\Fold2_F_PT_model.pth"
    finetuned_uni_checkpoint = r"C:\Users\Vivian\Documents\CONCH\_finetune_weights_UNI\linprob_ann_CL_uni.pth"
    train_csv = r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\train_patches.csv"
    val_csv = r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\val_patches.csv"
    test_csv = r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\test_patches.csv"

    # WandB settings
    use_wandb = True
    wandb_project = "resnet50_finetuning"
    wandb_run_name = "linprob_2.5x"
    wandb_config = {
        "epochs": 5,
        "lr": 2e-5,
        "model": "ResNet50",
        "task": "linear_probing",
        "magnification": "2.5x"
    }

    # Logging & output
    model_dir = r"C:\Users\Vivian\Documents\CONCH\_finetune_weights_ResNet50\wandb_2.5x"
    csv_dir = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\wandb_2.5x"
    model_filename = "linprob.pth"
    prediction_filename = "ResNet50_linprob.csv"
