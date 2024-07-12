from typing import Optional, Tuple, Union
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model,HubertPreTrainedModel,WavLMPreTrainedModel,HubertModel, WavLMModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn

from model.grl import *
from module.grl_model_outputs import GRLModelOutputs

_HIDDEN_STATES_START_POSITION = 2

class GRLClassification(Wav2Vec2PreTrainedModel, WavLMPreTrainedModel, HubertPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        # self.wav2vec2 = Wav2Vec2Model(config)
        self.hubert = HubertModel(config)
        # self.wavlm = WavLMModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        
        self.speaker_classifier = None
        self.speaker_lamda = 0

        # Initialize weights and apply final processing
        self.post_init()
    
    def init_lamda(self, lamda):
        self.speaker_lamda = lamda

    def init_speaker(self, num_speaker):
        self.speaker_classifier = GRLClassifier(self.config.classifier_proj_size, num_speaker)
        self.config.num_speaker = num_speaker

    def freeze_layers(self, num_layers_to_freeze: int):
        for param in self.wav2vec2.encoder.layers[num_layers_to_freeze:].parameters():
            param.requires_grad = False
        
    
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()
        
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
    ) -> Tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        speaker_loss = None
        speaker_logits = None
        if speaker_labels is not None:
            speaker_logits = self.speaker_classifier(pooled_output)
            loss_fct = nn.CrossEntropyLoss()
            speaker_loss = loss_fct(speaker_logits.view(-1, self.config.num_speaker), speaker_labels.view(-1))
            loss += self.speaker_lamda * speaker_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return GRLModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # domain_logits=speaker_logits
        )

    