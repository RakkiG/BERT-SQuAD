import torch.nn as nn

from models.BasicBert import BertModel


class BertQA(nn.Module):

    def __init__(self, args, config):
        super(BertQA, self).__init__()

        self.bert = BertModel.from_pretrained(config, args.bert_dir)
        self.qa_outputs = nn.Linear(args.hidden_size, 2)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                start_positions=None,
                end_positions=None):
        """
        :param input_ids: [src_len,batch_size]
        :param attention_mask: [batch_size,src_len]
        :param token_type_ids: [src_len,batch_size]
        :param position_ids:
        :param start_positions: [batch_size]
        :param end_positions:  [batch_size]
        :return:
        """
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1] # the output of the last layer of Bert
        # sequence_output: [src_len, batch_size, hidden_size]
        logits = self.qa_outputs(sequence_output)  # [src_len, batch_size,2]
        start_logits, end_logits = logits.split(1, dim=-1)
        # [src_len,batch_size,1]  [src_len,batch_size,1]
        start_logits = start_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        end_logits = end_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            return (start_loss + end_loss) / 2.0, start_logits, end_logits
        else:
            return sequence_output, start_logits, end_logits  # [batch_size,src_len]
