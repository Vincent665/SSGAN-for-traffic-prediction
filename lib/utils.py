import numpy as np
import torch
import torch.utils.data


def compute_val_loss(net, val_loader, criterion, sw, epoch, limit=None):
    DEVICE = torch.device('cuda:0')
    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, (encoder_inputs, labels) in enumerate(val_loader):

            encoder_inputs = encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = net(encoder_inputs)

            loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 50 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def predict_and_save_results(net, data_loader):
    DEVICE = torch.device('cuda:0')
    net.train(False)  # ensure dropout layers are in test mode
    preds = []
    trues = []

    with torch.no_grad():

        loader_length = len(data_loader)  # nb of batch

        input = []  # 存储所有batch的input

        for batch_index, (encoder_inputs, labels) in enumerate(data_loader):

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)
            encoder_inputs = encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = net(encoder_inputs)

            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            pred = outputs
            true = labels
            preds.append(pred)
            trues.append(true)

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse = metric(preds, trues)
    print('rmse:{},mse:{},mae:{}'.format(rmse, mse, mae))

    return


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)

    return mae, mse, rmse


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 3) // 2))}
    if epoch in lr_adjust.keys() and epoch > 3:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, net, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(net.state_dict(), path)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(net.state_dict(), path)
            print("save model")
            self.counter = 0
