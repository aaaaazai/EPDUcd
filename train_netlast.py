import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from dataset import load_img
import os
import torch.nn.functional as F
from models.model import Network
from models.loss import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger, Timer
import time


class CDTrainer():

    def __init__(self, train_dataloader, val_dataloader, batch_size):

        self.train_dataloader = train_dataloader  # 训练集
        self.val_dataloader = val_dataloader  # 验证集

        self.n_class = 2
        self.net_G = Network()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_G = self.net_G.to(self.device)

        print(self.device)
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.exp_lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=20, gamma=0.7)
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.checkpoint_dir = "test_1/"
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)

        self.timer = Timer()
        self.batch_size = batch_size
        self.begin_total = time.time()
        self.end_total = 0
        self.epoch_mf1 = 0
        self.best_val_f1 = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = 200
        self.global_step = 0
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
        self.G_pred = None
        self.batch = None
        self.G_loss = 0
        self.G_loss1 = 0
        self.G_loss2 = 0
        self.G_loss3 = 0
        self.G_loss4 = 0
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0

        self.loss1 = TLoss()
        self.loss2 = BinaryCombinedLoss()
        self.loss3 = BCEFocalLoss()
        self.loss5 = MyEntropyLoss()

    def _load_checkpoint(self, ckpt_name='last.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('加载最近的检查点...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name), map_location=self.device,weights_only=False)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(self.device)
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_f1 = checkpoint['best_val_f1']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('从批次%d开始, 最好f1 = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_f1, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('从头开始训练')
    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_f1': self.best_val_f1,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self, labels):
        target = labels.to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred_sigmoid = torch.sigmoid(G_pred)
        G_pred_bin = (G_pred_sigmoid >= 0.5).int()
        G_pred_np = G_pred_bin.cpu().numpy().astype(np.int64)

        target_np = target.cpu().numpy().astype(np.int64)
        current_score = self.running_metric.update_cm(pr=G_pred_np, gt=target_np)
        return current_score

    def _collect_running_batch_states(self, labels):
        running_f1 = self._update_metric(labels)
        m = len(self.train_dataloader)
        if self.is_training is False:
            m = len(self.val_dataloader)
        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = '是否是训练: %s. [%d,%d][%d,%d], 每秒实例: %.2f, 剩余时间: %.2fh, 总损失: %.5f, running_mf1: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est, self.G_loss.item(), running_f1)
            self.logger.write(message)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_mf1 = scores['F1_1']
        self.logger.write('是否训练: %s. 批次 %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_mf1))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def _update_checkpoints(self):
        self._save_checkpoint(ckpt_name='last.pt')
        self.logger.write('最近一次模型更新. Epoch_mf1=%.4f, Historical_best_f1=%.4f (at epoch %d)\n'
                          % (self.epoch_mf1, self.best_val_f1, self.best_epoch_id))
        self.logger.write('\n')
        if self.epoch_mf1 > self.best_val_f1:
            self.best_val_f1 = self.epoch_mf1
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best.pt')
            self.logger.write('*' * 20 + '最好的模型更新啦！！！！！！！！！！！！！！！！！！！！！！！\n' + '*' * 20)
            self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, T1, T2):
        T1 = T1.to(self.device)
        T2 = T2.to(self.device)
        self.T_prime, self.Delta_prime, self.grad_Delta_prime, self.f2e, self.f3e, self.changemap = self.net_G(T1=T1, T2=T2)
        self.G_pred = self.changemap

    def _backward_G(self, T1, labels, grad_label):
        T1 = T1.to(self.device)
        grad_label = grad_label.to(self.device)
        labels = labels.to(self.device)

        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        grad_label[grad_label >= 0.5] = 1
        grad_label[grad_label < 0.5] = 0

        self.G_loss1 = self.loss1(self.T_prime, T1)
        self.G_loss2 = self.loss2(self.G_pred, labels)

        self.G_loss3 = (self.loss3(self.f2e, grad_label) +
                        self.loss3(self.f3e, grad_label) +
                        self.loss3(self.grad_Delta_prime, grad_label)
                        )

        self.G_loss = self.G_loss1 + 5 * self.G_loss2 + 5 * self.G_loss3

        self.G_loss.backward()


    def train_models(self):

        print(self.steps_per_epoch)
        print(self.total_steps)

        self._load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self.logger.write('\nEpoch: %d' % (self.epoch_id + 1))
            end = time.time()
            hua = end - self.begin_total
            self.logger.write(
                f'\nafter：{hua // 3600} Hour {(hua % 3600) // 60} Minute {int(hua % 60)} second ')
            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            self.logger.write('\nlr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])

            for self.batch_id, (T1, T2, labels, grad_label) in enumerate(self.train_dataloader, 0):

                self._forward_pass(T1, T2)
                self.optimizer_G.zero_grad()
                self._backward_G(T1, labels, grad_label)
                self.optimizer_G.step()
                self._collect_running_batch_states(labels)

            self._collect_epoch_states()

            self._update_lr_schedulers()

            self.logger.write('评估开始...\n')

            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            for self.batch_id, (T1, T2, labels, grad_label) in enumerate(self.val_dataloader, 0):
                with torch.no_grad():
                    self._forward_pass(T1, T2)
                self._collect_running_batch_states(labels)


            self._collect_epoch_states()
            self._timer_update()

            self._update_checkpoints()

            end_these = time.time()
            hua1 = end_these - end
            self.logger.write(
                f'\nthis epoch time：{hua1 // 3600} hour {(hua1 % 3600) // 60} minute {int(hua1 % 60)} second\n')
        self.end_total = time.time()  # 总训练完的绝对时间
        hua2 = self.end_total - self.begin_total
        self.logger.write(
            f'\ntotal time：{hua2 // 3600} hour {(hua2 % 3600) // 60} minute {int(hua2 % 60)} second\n')   # 总训练完的相对时间


if __name__ == '__main__':
    train_dataloader, test_dataloader, val_dataloader, batch_size = load_img('./datasets/LEVIRCD_FC')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CDTrainer(train_dataloader, val_dataloader, batch_size)
    model.train_models()
