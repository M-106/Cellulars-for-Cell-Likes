# ----------
# > Import <
# ----------
import os
import shutil

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

    # first save the other file
    latest_train_state_file_path = os.path.join(output_dir, "last_train_state.pth")
    latest_train_state_2_file_path = os.path.join(output_dir, "last_train_state_2.pth")

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(latest_train_state_file_path):
        if os.path.exists(latest_train_state_2_file_path):
            os.remove(latest_train_state_2_file_path)
        shutil.move(latest_train_state_file_path, latest_train_state_2_file_path)

    torch.save(checkpoint, os.path.join(output_dir, "last_train_state.pth"))



def load_train_state(output_dir, model, optimizer, scheduler):
    latest_train_state_file_path = os.path.join(output_dir, "last_train_state.pth")
    latest_train_state_2_file_path = os.path.join(output_dir, "last_train_state_2.pth")

    for cur_path in [latest_train_state_file_path, latest_train_state_2_file_path]:
        try:
            if not os.path.exists(cur_path):
                raise ValueError(f"Cannot find following path/file: {cur_path}")
            checkpoint = torch.load(cur_path, map_location='cpu')

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
        except Exception as e:
            pass

    raise RuntimeError(f"Could not find/load any of the 2 train states ('{latest_train_state_file_path}', '{latest_train_state_2_file_path}').\n And following error occured: {e}")



def get_config_and_dir_from_last_train(output_dir):
    latest_train_state_file_path = os.path.join(output_dir, "last_train_state.pth")
    latest_train_state_2_file_path = os.path.join(output_dir, "last_train_state_2.pth")

    for cur_path in [latest_train_state_file_path, latest_train_state_2_file_path]:
        try:
            if not os.path.exists(cur_path):
                raise ValueError(f"Cannot find following path/file: {cur_path}")
            checkpoint = torch.load(cur_path, map_location='cpu')
            return checkpoint['config'], checkpoint['output_dir']
        except Exception as e:
            pass
    
    raise RuntimeError(f"Could not find/load any of the 2 train states ('{latest_train_state_file_path}', '{latest_train_state_2_file_path}').\n And following error occured: {e}")










