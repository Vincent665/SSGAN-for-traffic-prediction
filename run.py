import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import shutil
import argparse
import configparser
from model.Generator import make_model, NdMamba2_1d, DataEmbedding_inverted
from lib.utils import compute_val_loss, predict_and_save_results, EarlyStopping, adjust_learning_rate
from data_provider.data_factory import data_provider
from tensorboardX import SummaryWriter
import random
import torch.nn.functional as F
import torch.fft as fft
import ast


# ======================== SSGAN 判别器 ========================

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dropout):
        super().__init__()
        # 动态生成不同卷积分支
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs = [F.leaky_relu(conv(x)) for conv in self.convs]
        out = torch.cat(outs, dim=1)  # 按通道拼接
        return self.dropout(out)


class Discriminator(nn.Module):
    def __init__(self, pred_len, channels, num_of_vertices, kernel_sizes, dropout):
        super().__init__()
        self.pred_len = pred_len // 2 + 1
        N = num_of_vertices  # 输入通道数 = 输出通道数

        # 多尺度卷积：输入 2*N，输出 len(kernel_sizes)*channels
        self.multi_scale = MultiScaleConv(2 * N, channels, kernel_sizes, dropout)

        # 压缩回 N
        self.proj = nn.Conv1d(len(kernel_sizes) * channels, N, kernel_size=1)

        # 全连接分类层
        self.fc = nn.Linear(self.pred_len, 1)

    def forward(self, x):
        B, T, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, T)
        x_fft = torch.fft.rfft(x, dim=-1)
        real, imag = x_fft.real, x_fft.imag
        x_fft = torch.stack([real, imag], dim=-2)  # (B, N, 2, T)
        x_fft = x_fft.reshape(B, 2 * N, -1)  # (B, 2*N, T//2+1)
        out = self.multi_scale(x_fft)  # [B, len(kernel_sizes)*channels, seq_len]
        out = self.proj(out)  # [B, N, seq_len]
        out = self.fc(out).reshape(B, N)  # [B, N]
        return out


# ======================== 数据获取函数 ========================
def _get_data(root_path, flag, seq_len, label_len, pred_len, batch_size, dataset_name):
    data_set, data_loader = data_provider(root_path, flag, seq_len, label_len, pred_len, batch_size, dataset_name)
    return data_set, data_loader


# ======================== SSGAN 训练函数 ========================
def train_main(generator, discriminator, train_loader, val_loader, test_loader, test_data, params_path,
               start_epoch, epochs, criterion, g_optimizer, d_optimizer, sw, learning_rate, DEVICE, dataset_name,
               adv_weight):
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        sw.close()
        shutil.rmtree(params_path, ignore_errors=True)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('Resuming training from %s' % (params_path))
    else:
        raise SystemExit('Path error.')

    early_stopping = EarlyStopping(patience=3)
    best_val_loss = np.inf
    best_epoch = 0
    global_step = 0
    time_now = time.time()
    bce_loss = nn.BCEWithLogitsLoss()

    if start_epoch > 0:
        generator.load_state_dict(torch.load(os.path.join(params_path, 'G_epoch_%s.params' % start_epoch)))
        discriminator.load_state_dict(torch.load(os.path.join(params_path, 'D_epoch_%s.params' % start_epoch)))
        print('start epoch:', start_epoch)

    for epoch in range(start_epoch, epochs):
        iter_count = 0
        generator.train()
        discriminator.train()
        for batch_index, (encoder_inputs, labels) in enumerate(train_loader):
            iter_count += 1
            encoder_inputs = encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            ###### 1. Train Discriminator ######
            with torch.no_grad():
                fake_outputs = generator(encoder_inputs)

            d_optimizer.zero_grad()
            real_preds = discriminator(labels)
            fake_preds = discriminator(fake_outputs.detach())

            d_loss_real = bce_loss(real_preds, torch.ones_like(real_preds))
            d_loss_fake = bce_loss(fake_preds, torch.zeros_like(fake_preds))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            ###### 2. Train Generator ######
            g_optimizer.zero_grad()
            gen_outputs = generator(encoder_inputs)
            pred_loss = criterion(gen_outputs, labels)
            adv_loss = bce_loss(discriminator(gen_outputs), torch.ones_like(real_preds))
            clamped_adv_weight = torch.clamp(adv_weight, min=1e-5, max=1.0)
            g_loss = pred_loss + clamped_adv_weight * adv_loss  # 控制对抗性损失的权重
            g_loss.backward()
            g_optimizer.step()

            global_step += 1
            sw.add_scalar('G_loss', g_loss.item(), global_step)
            sw.add_scalar('D_loss', d_loss.item(), global_step)

            if global_step % 100 == 0:
                print("[{}/{}][{}/{}] G_loss: {:.4f}, D_loss: {:.4f}".format(
                    epoch, epochs, batch_index, len(train_loader), g_loss.item(), d_loss.item()))
            if (batch_index + 1) % 300 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_index + 1, epoch + 1, g_loss.item()))
                speed = (time.time() - time_now) / iter_count
                print("speed:", speed)
                allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                total = allocated_memory + cached_memory
                print('allocated_memory:', allocated_memory)
                print('cached_memory:', cached_memory)
                print('total:', total)
                iter_count = 0
                time_now = time.time()

        val_loss = compute_val_loss(generator, val_loader, criterion, sw, epoch, DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        early_stopping(val_loss, generator, os.path.join(params_path, 'G_epoch_{}.params'.format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(params_path, 'D_epoch_{}.params'.format(epoch)))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(g_optimizer, epoch + 1, learning_rate)
        adjust_learning_rate(d_optimizer, epoch + 1, learning_rate)

    print("Best epoch:", best_epoch)
    predict_main(best_epoch, test_loader, params_path, generator)


# ======================== 预测函数 ========================
def predict_main(global_step, data_loader, params_path, net):
    params_filename = os.path.join(params_path, 'G_epoch_{}.params'.format(global_step))
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results(net, data_loader)


# ======================== 固定种子 ========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/PEMS04_mgcn.conf', type=str, help="configuration file path")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    print('Read configuration file: %s' % (args.config))

    data_config = config['Data']
    training_config = config['Training']

    dataset_filename = data_config['dataset_filename']
    num_of_vertices = int(data_config['num_of_vertices'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']
    ctx = training_config['ctx']
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()
    print("CUDA:", USE_CUDA, DEVICE)
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    batch_size = int(training_config['batch_size'])
    loss_function = training_config['loss_function']
    use_norm = eval(training_config['use_norm'])
    dmodel = int(training_config['dmodel'])
    DataEmbedding = int(training_config['DataEmbedding'])
    features = int(training_config['features'])
    d_state = int(training_config['d_state'])
    DropoutG = float(training_config['DropoutG'])
    DropoutD = float(training_config['DropoutD'])
    num_channels = int(training_config['num_channels'])
    kernel_sizes = training_config['kernel_sizes']
    if isinstance(kernel_sizes, str):
        kernel_sizes = ast.literal_eval(kernel_sizes)
    folder_dir = "gan_predict{}".format(num_for_predict)
    params_path = os.path.join('experiments', dataset_name, folder_dir)

    train_data, train_loader = _get_data(dataset_filename, 'train', len_input, 0, num_for_predict, batch_size, dataset_name)
    val_data, val_loader = _get_data(dataset_filename, 'val', len_input, 0, num_for_predict, batch_size, dataset_name)
    test_data, test_loader = _get_data(dataset_filename, 'test', len_input, 0, num_for_predict, batch_size, dataset_name)

    generator = make_model(DEVICE, num_for_predict, len_input, use_norm, dmodel, DataEmbedding, DropoutG,
                           num_of_vertices, features, d_state)
    discriminator = Discriminator(pred_len=num_for_predict, channels=num_channels, num_of_vertices=num_of_vertices,
                                  kernel_sizes=kernel_sizes, dropout=DropoutD)

    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    criterion = nn.L1Loss() if loss_function == 'mae' else nn.MSELoss()
    criterion = criterion.to(DEVICE)

    adv_weight = torch.nn.Parameter(torch.tensor(0.01, device=DEVICE, requires_grad=True))
    g_optimizer = optim.Adam(list(generator.parameters()) + [adv_weight], lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    sw = SummaryWriter(logdir=params_path)

    train_main(generator, discriminator, train_loader, val_loader, test_loader, test_data, params_path,
               start_epoch, epochs, criterion, g_optimizer, d_optimizer, sw, learning_rate, DEVICE, dataset_name,
               adv_weight)


if __name__ == '__main__':
    set_seed(2025)
    main()
