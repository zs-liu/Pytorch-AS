import torch.cuda as cuda


def get_memory_use():
    device = cuda.current_device()
    message = cuda.get_device_name(device) + ':\n'
    message += 'allocated:' + str(cuda.memory_allocated(device)) + '/' + str(cuda.max_memory_allocated()) + '\n'
    message += 'cached:' + str(cuda.memory_cached(device)) + '/' + str(cuda.max_memory_cached()) + '\n'
    return message
