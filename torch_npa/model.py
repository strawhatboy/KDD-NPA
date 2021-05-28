
import torch
import torch.nn as nn
from consts import K
import logging

logger = logging.getLogger('model')


class NPAModel(nn.Module):
    def __init__(self, user_count, word_count, embedding_mat):
        super().__init__()
        self.user_count = user_count
        self.word_count = word_count
        self.user_embedding = nn.Embedding(user_count, 50)
        self.user_embedding_ = nn.Sequential(
            nn.Linear(50, 200),
            nn.Flatten()
        )
        if embedding_mat is not None:
            self.embedding_mat = torch.Tensor(embedding_mat)
            self.news_embedding = nn.Sequential(
                # _weight=embedding_matrix), output dim=4
                nn.Embedding(word_count, 300, _weight=self.embedding_mat),
                nn.Dropout(0.2),
            )
        else:
            self.news_embedding = nn.Sequential(
                nn.Embedding(word_count, 300),
                nn.Dropout(0.2),
            )
        # nn.Conv1d(30, 400, 3, 1, padding_mode='zeros'),

        # 不对，不是这么卷的
        # nn.Conv2d(50, 400, (3, 1), 1, padding_mode='zeros'),

        # 也不是这么卷的
        # nn.Conv2d(1, 400, (3, 300), 1, (1, 0)),
        self.news_conv = nn.Sequential(
            nn.Conv1d(300, 400, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention_tanh_layer = nn.Sequential(
            nn.Linear(200, 400),
            nn.Tanh(),
        )
        self.attention_news_tanh_layer = nn.Sequential(
            nn.Linear(320, 400),
            nn.Tanh(),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def get_user_embedding(self, user_input: torch.Tensor):
        user_embedding = self.user_embedding(user_input)
        user_embedding_word = self.user_embedding_(user_embedding)
        user_embedding_news = self.user_embedding_(user_embedding)

        return user_embedding_news, user_embedding_word

    def get_news_rep(self, user_embedding_word: torch.Tensor, news_input: torch.Tensor) -> torch.Tensor:
        # [batch, sentance, word, word_embedding]
        news_embedding = self.news_embedding(news_input)

        # [batch, word_embedding(300), sentance, word]
        news_embedding = news_embedding.permute(0, 3, 1, 2)

        # [batch, word_embedding(400), sentance, word]
        news_embedding_conv = self.news_conv(news_embedding)

        # [batch, user_embedding(400)]
        user_embedding_word_after_tanh = self.attention_tanh_layer(
            user_embedding_word)
        logger.debug('news_embedding_conv shape: {}, user_embedding_word_after_tanh shape: {}'.format(
            news_embedding_conv.shape, user_embedding_word_after_tanh.shape))

        # [batch, sentance, word]
        attention_a = torch.einsum(
            'ijkl,ij->ikl', news_embedding_conv, user_embedding_word_after_tanh)
        # attention_a = torch.bmm(news_embedding_conv,
        # user_embedding_word_after_tanh)
        logger.debug('attention_a shape: {}'.format(attention_a.shape))
        attention_weight = self.softmax(attention_a)

        # news_rep = torch.dot(news_embedding_conv, attention_weight)
        # [batch, embedding(400), sentance]
        news_rep = torch.einsum(
            'ijkl,ikl->ijk', news_embedding_conv, attention_weight)
        logger.debug('news_rep shape: {}'.format(news_rep.shape))
        return news_rep

    def get_user_rep(self, user_embedding_word: torch.Tensor, user_embedding_news: torch.Tensor, news_inputs: torch.Tensor) -> torch.Tensor:
        # tensor op, no need to expand_dim and concatenate like the TF version
        # news_reps = news_inputs.apply_(lambda x: self.get_news_rep(user_embedding_word, x))

        # [batch, news_embedding(400), sentance(news)]
        news_reps = self.get_news_rep(user_embedding_word, news_inputs)

        # [batch, user_embedding(400)]
        user_embedding_news_after_tanh = self.attention_tanh_layer(
            user_embedding_news)

        # [batch, sentances(news)]
        attention_news = torch.einsum(
            'ijk,ij->ik', news_reps, user_embedding_news_after_tanh)
        # attention_news = torch.bmm(
        #     news_reps, user_embedding_news_after_tanh)
        attention_weight = self.softmax(attention_news)

        # [batch, embedding(400)]
        user_rep = torch.einsum('ijk,ik->ij', news_reps, attention_weight)
        logger.debug('user_rep shape: {}'.format(user_rep.shape))
        # user_rep = torch.dot(news_reps, attention_weight)
        return user_rep

    def predict_candidates(self, user_embedding_word: torch.Tensor, user_rep: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:

        # [batch, embedding(400), sentance]
        candidates_rep = self.get_news_rep(user_embedding_word, candidates)

        logits = torch.einsum('ijk,ij->ik', candidates_rep, user_rep)
        # logits = candidates_rep.apply_(lambda x: torch.dot(user_rep, x))
        logits = self.softmax(logits)
        logger.debug('logits shape: {}'.format(logits.shape))
        return logits

    def predict_candidate(self, user_embedding_word: torch.Tensor, user_rep: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        candidate_rep = self.get_news_rep(
            user_embedding_word, candidate.unsqueeze(1))
        logit = torch.einsum('ij,ij->i', candidate_rep.squeeze(-1), user_rep)
        logit = self.sigmoid(logit)
        logger.debug('logit shape: {}'.format(logit.shape))
        return logit

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        user = x
        history = y
        candidates = z

        logger.debug('forwarding train: user: {}, history: {}, candidates: {}'.format(
            user.shape, history.shape, candidates.shape))

        user_embedding_word, user_embedding_news = self.get_user_embedding(
            user)
        user_rep = self.get_user_rep(
            user_embedding_word, user_embedding_news, history)
        logits = self.predict_candidates(
            user_embedding_word, user_rep, candidates)
        return logits

    def forward_test(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        # 好像不能这么写，因为这tensor中的元素维度不同，那该怎么办？那就穿多个参数！
        user = x
        history = y
        candidate = z

        user_embedding_word, user_embedding_news = self.get_user_embedding(
            user)
        user_rep = self.get_user_rep(
            user_embedding_word, user_embedding_news, history)
        logit = self.predict_candidate(
            user_embedding_word, user_rep, candidate)
        return logit
