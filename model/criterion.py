import torch.nn as nn
import torch.nn.functional as F

class CRITERION(nn.Module):
    def __init__(self, config, logger):
        super(CRITERION, self).__init__()
        self.config = config
        self.logger = logger
        self.c_fusion = nn.CrossEntropyLoss() # mean default

        # every loss weight
        self.major_loss_weight = float(self.config['hyper_para']['loss_weight']['major_loss_weight'])
        self.distribution_loss_weight = float(self.config['hyper_para']['loss_weight']['distribution_loss_weight'])
        self.euclidean_distance_fusion_q_weight = float(self.config['hyper_para']['loss_weight']['euclidean_distance_fusion_q_weight'])
        self.euclidean_distance_fusion_a_weight = float(self.config['hyper_para']['loss_weight']['euclidean_distance_fusion_a_weight'])
        self.euclidean_distance_fusion_v_weight = float(self.config['hyper_para']['loss_weight']['euclidean_distance_fusion_v_weight'])

        self.cycle_Kl_loss_weight = float(self.config['hyper_para']['loss_weight']['cycle_Kl_loss_weight'])
        self.cycle_KL_a_q_weight = float(self.config['hyper_para']['loss_weight']['cycle_KL_a_q_weight'])
        self.cycle_KL_q_v_weight = float(self.config['hyper_para']['loss_weight']['cycle_KL_q_v_weight'])
        self.cycle_KL_v_a_weight = float(self.config['hyper_para']['loss_weight']['cycle_KL_v_a_weight'])

        self.cycle_KL_loss = None
        self.euclidean_distance = None
        if self.config['hyper_para']['bias_learner']['three_bias_learner_exist']:
            self.cycle_KL_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
            self.euclidean_distance = nn.PairwiseDistance(p=2)

    def forward(self, output, label):
        # get every logits
        output_fusion_logits = output['fusion_logits']
        output_q_bias_logits = output['q_bias_logits']
        output_a_bias_logits = output['a_bias_logits']
        output_v_bias_logits = output['v_bias_logits']

        label = label.view(-1) # [batch_size 1] -> [1 batch_size]

        fusion_loss = self.c_fusion(input=output_fusion_logits.cuda(), target=label.cuda())

        loss = self.major_loss_weight * fusion_loss

        if self.config['hyper_para']['bias_learner']['three_bias_learner_exist']:
            # to max the distribution between a specific channel and fusion

            # 不同分母版本
            euclidean_distance_fusion_q_loss = self.euclidean_distance(output_fusion_logits, output_q_bias_logits).mean()
            euclidean_distance_fusion_a_loss = self.euclidean_distance(output_fusion_logits, output_a_bias_logits).mean()
            euclidean_distance_fusion_v_loss = self.euclidean_distance(output_fusion_logits, output_v_bias_logits).mean()
            distribution_loss = self.euclidean_distance_fusion_q_weight / (euclidean_distance_fusion_q_loss + 0.00001)
            distribution_loss += self.euclidean_distance_fusion_a_weight / (euclidean_distance_fusion_a_loss + 0.00001)
            distribution_loss += self.euclidean_distance_fusion_v_weight / (euclidean_distance_fusion_v_loss + 0.00001)
            distribution_loss *= self.distribution_loss_weight

            # 相同分母版本
            # euclidean_distance_fusion_q_loss = self.euclidean_distance_fusion_q_weight * self.euclidean_distance(output_fusion_logits, output_q_bias_logits).mean()
            # euclidean_distance_fusion_a_loss = self.euclidean_distance_fusion_a_weight * self.euclidean_distance(output_fusion_logits, output_a_bias_logits).mean()
            # euclidean_distance_fusion_v_loss = self.euclidean_distance_fusion_v_weight * self.euclidean_distance(output_fusion_logits, output_v_bias_logits).mean()
            # distribution_loss = self.distribution_loss_weight / (euclidean_distance_fusion_q_loss + euclidean_distance_fusion_a_loss + euclidean_distance_fusion_v_loss)

            loss += distribution_loss

            # to min the cycle distribution between two channels, see nn.KLDivLoss torch official doc to get more detail
            audio_question_KL_loss = self.cycle_KL_loss(input=F.log_softmax(output_a_bias_logits.cuda(),dim=1), target=F.softmax(output_q_bias_logits.cuda(),dim=1))
            question_visual_KL_loss = self.cycle_KL_loss(input=F.log_softmax(output_q_bias_logits.cuda(), dim=1), target=F.softmax(output_v_bias_logits.cuda(), dim=1))
            visual_audio_KL_loss = self.cycle_KL_loss(input=F.log_softmax(output_v_bias_logits.cuda(), dim=1), target=F.softmax(output_a_bias_logits.cuda(), dim=1))

            cycle_KL_loss = self.cycle_KL_a_q_weight * audio_question_KL_loss + self.cycle_KL_q_v_weight * question_visual_KL_loss + self.cycle_KL_v_a_weight * visual_audio_KL_loss
            cycle_KL_loss *= self.cycle_Kl_loss_weight

            loss += cycle_KL_loss

        return loss
