import os
import random
import torch
import numpy as np
from model.net import RAVQA
from dataset.factory import produce_split
from torch.utils.data import DataLoader
from utils import Logger
from model.criterion import CRITERION
from torch import optim


class Engine(object):
    def __init__(self, config):
        super(Engine, self).__init__()
        self.config = config
        self.set_run_envir()
        self.logger = Logger(config).get_log()
        self.model = RAVQA(config).cuda()
        self.dataset = produce_split(config, self.logger)

    def get_optimizer_and_scheduler(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.config['hyper_para']['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['hyper_para']['scheduler_step'],
                                              gamma=self.config['hyper_para']['scheduler_gamma'])
        return optimizer, scheduler

    def train_one_epoch(self, epoch, train_data_loader, optimizer, criterion):
        self.model.train()
        for batch_index, batch_inputs in enumerate(train_data_loader):
            optimizer.zero_grad() #
            output = self.model(batch_inputs)
            loss = criterion(output=output, label=batch_inputs['labels'])
            loss.backward()
            optimizer.step()

            # log train message
            if batch_index % self.config['log_interval'] == 0:
                log_string = 'Epoch {} complete [{:.0f}%]\tLoss: {:.6f}'.format(epoch, 100.0 * batch_index / len(train_data_loader), loss.item())
                self.logger.info(msg=log_string)

    def train(self):
        self.logger.info(msg='Begin to train.')
        self.model.train()
        # 统计可训练参数数量
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(msg='The number of trainable parameters is {}'.format(total_trainable_params))
        train_data_loader = DataLoader(dataset=self.dataset['train_split'], batch_size=self.config['hyper_para']['batch_size'], shuffle=True, num_workers=self.config['hyper_para']['train_num_workers'])
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        criterion = CRITERION(self.config, self.logger)
        best_val_acc = 0
        for epoch in range(1, self.config['hyper_para']['epochs'] + 1):
            self.logger.info(msg='Train epoch {} begin'.format(epoch))  # log
            self.train_one_epoch(epoch, train_data_loader, optimizer, criterion)
            scheduler.step(epoch)
            epoch_val_acc = self.val()
            if epoch_val_acc >= best_val_acc:
                self.logger.info(msg='Train epoch {} end, val get best acc {:.2f}%'.format(epoch, 100.0 * epoch_val_acc))  # log
                best_val_acc = epoch_val_acc
                torch.save(self.model.state_dict(), os.path.join(self.config['path']['save_model'], self.config['model_name'] + '.pt'))
        self.logger.info(msg='Train end, model saved and named as ' + self.config['model_name'] + '.pt')

    def test(self):
        if os.path.exists(os.path.join(self.config['path']['save_model'], self.config['model_name'] + '.pt')):
            self.model.load_state_dict(torch.load(os.path.join(self.config['path']['save_model'], self.config['model_name'] + '.pt')))
        self.model.eval()

        for test_mode in ['original', 'extend', 'extend-head', 'extend-tail']:
            if self.config['test_mode'][test_mode]:
                # define list to store test result
                audio_qa_counting = [] #
                audio_qa_comparative = [] #
                visual_qa_counting = [] #
                visual_qa_location = [] #
                av_qa_existential = [] #
                av_qa_counting = [] #
                av_qa_location = [] #
                av_qa_comparative = [] #
                av_qa_temporal = [] #
                
                self.logger.info(msg='[{}] Begin to test on [{}] split.'.format(self.config['model_name'], test_mode))
                test_data_loader = DataLoader(dataset=self.dataset['test_split'][test_mode], batch_size=self.config['hyper_para']['eval_batch_size'], shuffle=False, num_workers=self.config['hyper_para']['eval_num_workers'])
                correct, total = 0, 0
                with torch.no_grad():
                    for batch_index, batch_inputs in enumerate(test_data_loader):
                        output_fusion_logits = self.model(batch_inputs)['fusion_logits']
                        predict = output_fusion_logits.argmax(dim=1)

                        labels = batch_inputs['labels'].cuda().view(-1)

                        num = predict.shape[0]  # num of this input batch
                        type = batch_inputs['type']

                        correct = correct + (predict == labels).sum().item()
                        total = total + num

                        for input_index in range(num):
                            right_or_not = 0
                            if predict[input_index] == labels[input_index]:
                                right_or_not = 1
                            if type[0][input_index] == 'Audio':
                                if type[1][input_index] == 'Counting':
                                    audio_qa_counting.append(right_or_not)
                                elif type[1][input_index] == 'Comparative':
                                    audio_qa_comparative.append(right_or_not)
                            elif type[0][input_index] == 'Visual':
                                if type[1][input_index] == 'Counting':
                                    visual_qa_counting.append(right_or_not)
                                elif type[1][input_index] == 'Location':
                                    visual_qa_location.append(right_or_not)
                            elif type[0][input_index] == 'Audio-Visual':
                                if type[1][input_index] == 'Existential':
                                    av_qa_existential.append(right_or_not)
                                elif type[1][input_index] == 'Counting':
                                    av_qa_counting.append(right_or_not)
                                elif type[1][input_index] == 'Location':
                                    av_qa_location.append(right_or_not)
                                elif type[1][input_index] == 'Comparative':
                                    av_qa_comparative.append(right_or_not)
                                elif type[1][input_index] == 'Temporal':
                                    av_qa_temporal.append(right_or_not)

                # show test result
                self.logger.info(msg='audio_qa_counting acc: %.2f %%' % (100 * sum(audio_qa_counting) / len(audio_qa_counting)))
                self.logger.info(msg='audio_qa_comparative acc: %.2f %%' % (100 * sum(audio_qa_comparative) / len(audio_qa_comparative)))
                self.logger.info(msg='audio_qa avg. Accuracy: %.2f %%' % (100 * (sum(audio_qa_counting) + sum(audio_qa_comparative)) / (
                        len(audio_qa_counting) + len(audio_qa_comparative))))
                self.logger.info(msg='visual_qa_counting acc: %.2f %%' % (100 * sum(visual_qa_counting) / len(visual_qa_counting)))
                self.logger.info(msg='visual_qa_location acc: %.2f %%' % (100 * sum(visual_qa_location) / len(visual_qa_location)))
                self.logger.info(msg='visual_qa avg. acc: %.2f %%' % (100 * (sum(visual_qa_counting) + sum(visual_qa_location)) / (
                        len(visual_qa_counting) + len(visual_qa_location))))
                self.logger.info(msg='av_qa_existential acc: %.2f %%' % (100 * sum(av_qa_existential) / len(av_qa_existential)))
                self.logger.info(msg='av_qa_location acc: %.2f %%' % (100 * sum(av_qa_location) / len(av_qa_location)))
                self.logger.info(msg='av_qa_counting acc: %.2f %%' % (100 * sum(av_qa_counting) / len(av_qa_counting)))
                self.logger.info(msg='av_qa_comparative acc: %.2f %%' % (100 * sum(av_qa_comparative) / len(av_qa_comparative)))
                self.logger.info(msg='av_qa_temporal acc: %.2f %%' % (100 * sum(av_qa_temporal) / len(av_qa_temporal)))
                self.logger.info(msg='av_qa avg. acc: %.2f %%' % (100 * (sum(av_qa_counting) + sum(av_qa_location) + sum(av_qa_existential) + sum(av_qa_temporal)
                                    + sum(av_qa_comparative)) / ( len(av_qa_counting) + len(av_qa_location) + len(av_qa_existential) +
                                         len(av_qa_temporal) + len(av_qa_comparative))))

                self.logger.info(msg="[{}] On test split [{}], test-acc: {:.2f}%".format(self.config['model_name'], test_mode, 100.0 * (correct / total)))

    def val(self):
        self.model.eval()
        val_data_loader = DataLoader(dataset=self.dataset['val_split'], batch_size=self.config['hyper_para']['eval_batch_size'], shuffle=False, num_workers=self.config['hyper_para']['eval_num_workers'])
        correct, total = 0, 0
        with torch.no_grad():
            for batch_index, batch_inputs in enumerate(val_data_loader):
                output_fusion_logits = self.model(batch_inputs)['fusion_logits']
                predict = output_fusion_logits.argmax(dim=1)
                correct = correct + (predict == batch_inputs['labels'].cuda().view(-1)).sum().item()
                total = total + predict.shape[0]
        return correct / total

    def set_run_envir(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['hyper_para']['gpu']
        torch.manual_seed(self.config['hyper_para']['seed'])
        np.random.seed(self.config['hyper_para']['seed'])
        random.seed(self.config['hyper_para']['seed'])
