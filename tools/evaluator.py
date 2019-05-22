import torch


class Evaluator:

    def __init__(self, batch_size, model, loader):
        self.batch_size = batch_size
        self.model = model
        self.loader = loader

    def evaluate(self, epoch, accu_list, loss_list):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad:
            for batch, (data, _) in enumerate(self.loader):
                self.model.zero_grad()
                loss, accu = self.model(data[0], data[1], data[2:])
                total_loss += loss
                total_accuracy += accu
        total_loss /= len(self.loader)
        total_accuracy /= len(self.loader)
        accu_list.append(total_accuracy)
        loss_list.append(total_loss)
        print('Valid Epoch: {} \tAverage Loss: {:.4f}\tAverage Accuracy: {:.4f}%'
              .format(epoch, total_loss, total_accuracy * 100))
