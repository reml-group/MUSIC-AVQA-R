import torch
import torch.nn as nn
from transformers import VisualBertModel
from model.layer import MLP
from utils import grad_mul_const


class RAVQA(nn.Module):
    def __init__(self, config):
        super(RAVQA, self).__init__()
        # https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertModel
        self.config = config
        # self.visual_bert = VisualBertModel.from_pretrained("./vlbert")
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        # project the feature to 768 and 2048 to adapt visualbert encoder.
        self.audio_linear = nn.Linear(128, 768)
        self.video_linear = nn.Linear(512, 768)
        self.av_linear = nn.Linear(768, 2048)
        self.dropout = nn.Dropout(p=0.1)
        self.cls = nn.Linear(768, config['hyper_para']['num_labels'])

        # bias learner
        if config['hyper_para']['bias_learner']['q_bias']:
            self.q_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])
        if config['hyper_para']['bias_learner']['a_bias']:
            self.a_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])
        if config['hyper_para']['bias_learner']['v_bias']:
            self.v_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])

    def forward(self, b_inputs):
        inputs, position_ids = b_inputs['inputs'], b_inputs['pos'].cuda()
        audio_emb, video_emb = b_inputs['audio_emb'].cuda(), b_inputs['video_emb'].cuda()
        audio_token_type_ids, video_token_type_ids = b_inputs['audio_token_type'].cuda(), b_inputs[
            'video_token_type'].cuda()

        audio_emb_mid = self.audio_linear(audio_emb)
        audio_emb = self.av_linear(audio_emb_mid)
        video_emb_mid = self.video_linear(video_emb)
        video_emb = self.av_linear(video_emb_mid)
        av_feat = torch.cat([video_emb, audio_emb], dim=1)
        av_token_type_ids = torch.cat([video_token_type_ids, audio_token_type_ids], dim=1)

        inputs = {key: inputs[key].cuda() for key in inputs}
        inputs.update({
            'visual_embeds': av_feat,
            'visual_token_type_ids': av_token_type_ids,
            'position_ids': position_ids
        })
        outputs = self.visual_bert(**inputs)
        pooler_output = outputs.pooler_output
        fusion_logits = self.cls(self.dropout(pooler_output))

        q_bias_logits, a_bias_logits, v_bias_logits = None, None, None
        if self.config['hyper_para']['bias_learner']['q_bias']:
            q_bias_logits = self.get_bias_classifier_logits_q(inputs)
        if self.config['hyper_para']['bias_learner']['a_bias']:
            a_bias_logits = self.get_bias_classifier_logits_a(audio_emb_mid)
        if self.config['hyper_para']['bias_learner']['v_bias']:
            v_bias_logits = self.get_bias_classifier_logits_v(video_emb_mid)

        return {
            'fusion_logits': fusion_logits,
            'q_bias_logits': q_bias_logits,
            'a_bias_logits': a_bias_logits,
            'v_bias_logits': v_bias_logits
        }

    def get_bias_classifier_logits_q(self, inputs):
        que_emb = self.visual_bert(
            input_ids=inputs['input_ids'],
            position_ids=inputs['position_ids'],
            attention_mask=inputs['attention_mask']
        )
        #que_emb = grad_mul_const(que_emb.pooler_output, 0.0)
        q_bias_logits = self.q_bias(que_emb.pooler_output)
        return q_bias_logits

    def get_bias_classifier_logits_a(self, inputs):
        audio_emb = self.visual_bert(inputs_embeds=inputs)
        #audio_emb = grad_mul_const(audio_emb.pooler_output, 0.0)
        a_bias_logits = self.a_bias(audio_emb.pooler_output)
        return a_bias_logits

    def get_bias_classifier_logits_v(self, inputs):
        video_emb = self.visual_bert(inputs_embeds=inputs)
        #video_emb = grad_mul_const(video_emb.pooler_output, 0.0)
        v_bias_logits = self.v_bias(video_emb.pooler_output)
        return v_bias_logits
