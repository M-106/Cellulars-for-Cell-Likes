# ----------
# > Import <
# ----------
import os

import torch



# -----------------------------
# > Stepwise Helper Functions <
# -----------------------------
def save_cur_train_state(
        output_dir,
        config,
        model,
        optimizer,
        scheduler,
        cur_epoch,
        best_val_accuracy,
        best_model_epoch,
        latest_model_epoch,
        best_checkpoint_path
    ):
    checkpoint = {
        'epoch': cur_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'best_model_epoch': best_model_epoch,
        'latest_model_epoch': latest_model_epoch,
        'best_checkpoint_path': best_checkpoint_path,
        'output_dir': output_dir,
        'config': config
    }
    torch.save(checkpoint, os.path.join(output_dir, "last_train_state.pth"))



def load_train_state(output_dir, model, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(output_dir, "last_train_state.pth"))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_accuracy = checkpoint['best_val_accuracy']
    best_model_epoch = checkpoint['best_model_epoch']
    latest_model_epoch = checkpoint['latest_model_epoch']
    best_checkpoint_path = checkpoint['best_checkpoint_path']

    print(f"Checkpoint loaded. Starting at epoch {start_epoch}")
    return start_epoch, best_val_accuracy, best_model_epoch, latest_model_epoch, best_checkpoint_path



def get_config_and_dir_from_last_train(output_dir):
    checkpoint = torch.load(os.path.join(output_dir, "last_train_state.pth"))
    return checkpoint['config'], checkpoint['output_dir']








