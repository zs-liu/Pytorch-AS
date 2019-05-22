import torch


def save_checkpoint(save_dir, model, optimizer):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_dir)
    return True


def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir)
    return checkpoint
