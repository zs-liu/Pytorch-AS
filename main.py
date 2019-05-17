import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data

from dataset import InsuranceAnswerDataset, DataEmbedding
from model import Matcher
from tools import Trainer, Evaluator


def main():
    batch_size = 256
    dataset_size = 1500
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 0
    show_frq = 20
    negative_size = 40
    negative_expand = 2
    save_dir = '/cos_person/data/'

    dm = DataEmbedding()

    dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=negative_size, data_type='train')
    valid_dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=dataset_size - negative_size,
                                           data_type='valid')
    test_dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=dataset_size - negative_size,
                                          data_type='test', full_flag=True, full_size=27410)

    print(len(dataset))

    model = Matcher(batch_size=batch_size, embedding_dim=dm.embedding_dim, vocab_size=dm.embedding_size,
                    hidden_dim=150)

    embedding_matrix = torch.Tensor(dm.get_embedding_matrix())
    if torch.cuda.is_available():
        embedding_matrix = embedding_matrix.cuda()
        model = model.cuda()
    model.encoder.embedding.weight.data.copy_(embedding_matrix)

    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    train_accu_list = []
    train_loss_list = []
    valid_accu_list = []
    valid_loss_list = []

    trainer = Trainer(model=model, loader=train_loader, optimizer=optimizer, batch_size=batch_size,
                      data_size=len(train_loader), threshold_decay=True)
    valider = Evaluator(model=model, loader=valid_loader, batch_size=batch_size)
    for epoch in range(1, epochs + 1):
        print('Epoch {} start...'.format(epoch))
        trainer.train(epoch=epoch, show_frq=show_frq, accu_list=train_accu_list, loss_list=train_loss_list)
        valider.evaluate(epoch=epoch, accu_list=valid_accu_list, loss_list=valid_loss_list)
        torch.save(train_loss_list, save_dir + 'train_loss.pkl')
        torch.save(train_accu_list, save_dir + 'train_accu.pkl')
        if negative_size + negative_expand + 10 <= dataset_size and negative_expand > 0:
            negative_size += negative_expand
            dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=negative_size)
            train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            trainer.loader = train_loader
        if epoch <= 1:
            torch.save(model, save_dir + 'model.pkl')
        elif valid_accu_list[-1] > valid_accu_list[-2] \
                or (valid_accu_list[-1] == valid_accu_list[-2] and valid_loss_list[-1] < valid_loss_list[-2]):
            torch.save(model, save_dir + 'model.pkl')
        else:
            model = torch.load(save_dir + 'model.pkl')
            trainer.model = model
            trainer._lr_decay(0.8)
            valider.model = model

    torch.save(train_loss_list, save_dir + 'train_loss.pkl')
    torch.save(train_accu_list, save_dir + 'train_accu.pkl')
    torch.save(valid_loss_list, save_dir + 'valid_loss.pkl')
    torch.save(valid_accu_list, save_dir + 'valid_accu.pkl')
    torch.save(model, save_dir + 'model.pkl')

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tester = Evaluator(model=model, loader=test_loader, batch_size=batch_size)
    test_accu_list = []
    test_loss_list = []
    tester.evaluate(epoch=1, accu_list=test_accu_list, loss_list=test_loss_list)
    torch.save(test_loss_list, save_dir + 'test_loss.pkl')
    torch.save(test_accu_list, save_dir + 'test_accu.pkl')


if __name__ == '__main__':
    main()
