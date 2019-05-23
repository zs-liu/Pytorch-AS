import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data

from dataset import InsuranceAnswerDataset, DataEmbedding
from model import Matcher
from tools import Trainer, Evaluator
from tools import save_checkpoint, load_checkpoint, get_memory_use


def main():
    batch_size = 64
    valid_batch_size = 8
    dataset_size = 500
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 30
    show_frq = 20
    negative_size = 10
    negative_expand = 1
    negative_size_bound = 20
    negative_retake = True
    load_read_model = False
    save_dir = '/cos_person/data/'
    torch.backends.cudnn.benchmark = True

    dm = DataEmbedding()

    dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=negative_size, data_type='train')
    valid_dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=400,
                                           data_type='valid')

    print(len(dataset))

    model = Matcher(embedding_dim=dm.embedding_dim, vocab_size=dm.embedding_size,
                    hidden_dim=150, tagset_size=50, negative_size=negative_size)

    embedding_matrix = torch.Tensor(dm.get_embedding_matrix())
    print('before model:' + get_memory_use())
    if torch.cuda.is_available():
        embedding_matrix = embedding_matrix.cuda()
        model = model.cuda()
    model.encoder.embedding.weight.data.copy_(embedding_matrix)
    print('after model:' + get_memory_use())

    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    train_accu_list = []
    train_loss_list = []
    valid_accu_list = []
    valid_loss_list = []

    trainer = Trainer(model=model, loader=train_loader, optimizer=optimizer, batch_size=batch_size,
                      data_size=len(train_loader), threshold_decay=True)
    valider = Evaluator(model=model, loader=valid_loader, batch_size=valid_batch_size)
    for epoch in range(1, epochs + 1):
        print('before:' + get_memory_use())
        print('Epoch {} start...'.format(epoch))
        model.reset_negative(dataset.negative_size)
        trainer.train(epoch=epoch, show_frq=show_frq, accu_list=train_accu_list, loss_list=train_loss_list)
        print('train after:' + get_memory_use())
        model.reset_negative(valid_dataset.negative_size)
        valider.evaluate(epoch=epoch, accu_list=valid_accu_list, loss_list=valid_loss_list)
        print('valid after:' + get_memory_use())
        torch.save(train_loss_list, save_dir + 'train_loss.pkl')
        torch.save(train_accu_list, save_dir + 'train_accu.pkl')
        if negative_retake:
            if negative_size + negative_expand <= negative_size_bound:
                negative_size += negative_expand
            del dataset
            del train_loader
            dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=negative_size)
            train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            trainer.loader = train_loader
        if epochs - epoch <= 5:
            load_read_model = True
        if load_read_model:
            if epoch <= 1:
                save_checkpoint(save_dir=save_dir + 'check.pkl', model=model, optimizer=optimizer)
            elif valid_accu_list[-1] > valid_accu_list[-2] \
                    or (valid_accu_list[-1] == valid_accu_list[-2] and valid_loss_list[-1] < valid_loss_list[-2]):
                save_checkpoint(save_dir=save_dir + 'check.pkl', model=model, optimizer=optimizer)
            else:
                checkpoint = load_checkpoint(save_dir + 'check.pkl')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.model = model
                trainer.optimizer = optimizer
                trainer._lr_decay(0.8)
                valider.model = model
        else:
            torch.save(model, save_dir + 'model.pkl')

    torch.save(train_loss_list, save_dir + 'train_loss.pkl')
    torch.save(train_accu_list, save_dir + 'train_accu.pkl')
    torch.save(valid_loss_list, save_dir + 'valid_loss.pkl')
    torch.save(valid_accu_list, save_dir + 'valid_accu.pkl')
    torch.save(model, save_dir + 'model.pkl')

    test_dataset = InsuranceAnswerDataset(dataset_size=dataset_size, negative_size=400, data_type='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=valid_batch_size, shuffle=True, drop_last=True)
    tester = Evaluator(model=model, loader=test_loader, batch_size=valid_batch_size)
    test_accu_list = []
    test_loss_list = []
    model.reset_negative(test_dataset.negative_size)
    tester.evaluate(epoch=1, accu_list=test_accu_list, loss_list=test_loss_list)
    torch.save(test_loss_list, save_dir + 'test_loss.pkl')
    torch.save(test_accu_list, save_dir + 'test_accu.pkl')


if __name__ == '__main__':
    main()
