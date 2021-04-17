import numpy
import torch
import pytorch_pretrained_bert


class MatchSum(torch.nn.Module):
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        from transformers import BertModel, RobertaModel

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text_id, candidate_id, summary_id):
        batch_size = text_id.size(0)

        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]  # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num,
                                          self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)  # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}


##########################################

class MatchSum_AnotherModule(pytorch_pretrained_bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(MatchSum_AnotherModule, self).__init__(config)
        self.embeddings = pytorch_pretrained_bert.modeling.BertEmbeddings(config)
        self.encoder = pytorch_pretrained_bert.modeling.BertEncoder(config)
        self.pooler = pytorch_pretrained_bert.modeling.BertPooler(config)
        self.margin = 1E-2
        self.apply(self.init_bert_weights)

    def __get_passage_vector(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.pooler(encoded_layers[0])
        return pooled_output

    def __get_loss(self, score, summary_score):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)

        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss += loss_func(pos_score, neg_score, ones)

        return TotalLoss

    def forward(self, passage_tokens, summary_tokens, candidate_tokens):
        passage_vector = self.__get_passage_vector(passage_tokens)
        summary_vector = self.__get_passage_vector(summary_tokens)
        candidate_vector = self.__get_passage_vector(candidate_tokens, attention_mask=~(candidate_tokens == 0))

        summary_score = torch.cosine_similarity(summary_vector, passage_vector, dim=-1)
        candidate_score = torch.cosine_similarity(candidate_vector, passage_vector, dim=-1).unsqueeze(0)
        loss = self.__get_loss(candidate_score, summary_score)
        return loss, candidate_score


class BertEmbeddingsExtend(torch.nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsExtend, self).__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings_extend = torch.nn.Embedding(num_embeddings=8000, embedding_dim=config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = pytorch_pretrained_bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.position_threshold = config.max_position_embeddings

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(min(seq_length, self.position_threshold), dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat([input_ids.size()[0], 1])

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if seq_length > self.position_threshold:
            position_ids_extend = torch.arange(self.position_threshold, seq_length, dtype=torch.long,
                                               device=input_ids.device)
            position_ids_extend = position_ids_extend.unsqueeze(0).repeat([input_ids.size()[0], 1])
            position_embeddings_extend = self.position_embeddings_extend(position_ids_extend)
            position_embeddings = torch.cat([position_embeddings, position_embeddings_extend], dim=1)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSumEncoder(pytorch_pretrained_bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertSumEncoder, self).__init__(config)
        self.embeddings = BertEmbeddingsExtend(config)
        self.encoder = pytorch_pretrained_bert.modeling.BertEncoder(config)
        # self.pooler = pytorch_pretrained_bert.modeling.BertPooler(config)
        self.predict_layer = torch.nn.Linear(in_features=768, out_features=2)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, text_tokens, mask_tokens, position_tokens, label=None):
        attention_mask = ~(text_tokens == 0)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(text_tokens, mask_tokens)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]

        sentence_hidden = []
        for indexX in range(len(position_tokens)):
            for indexY in range(len(position_tokens[indexX])):
                sentence_hidden.append(sequence_output[indexX][position_tokens[indexX][indexY]].unsqueeze(0))
        sentence_hidden = torch.cat(sentence_hidden, dim=0)
        sentence_predict = self.predict_layer(sentence_hidden)

        if label is not None:
            treat_label = []
            for sample in label:
                treat_label.extend(sample)
            assert len(treat_label) == len(sentence_predict)

            treat_label = torch.LongTensor(treat_label).cuda()
            loss = self.loss_function(input=sentence_predict, target=treat_label)
            return loss
        else:
            return sentence_predict
