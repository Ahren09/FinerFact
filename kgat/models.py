import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU

from utils.utils import fields_user
from utils.utils_misc import get_root_dir
from utils.utils_preprocess import load_user_embed_and_Rs


def kernal_mus(n_kernels):
    """
    get the mean mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels, sigma_val):
    assert n_kernels >= 1
    l_sigma = [0.001] + [sigma_val] * (n_kernels - 1)
    return l_sigma


class inference_model(nn.Module):
    def __init__(self, bert_model, args, config, tokenizer=None):
        super(inference_model, self).__init__()
        self.args = args
        self.use_cuda = args.cuda
        self.device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'
        self.bert_hidden_dim = args.bert_hidden_dim
        self.batch_size = args.train_batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        self.evi_num = args.evi_num
        self.nlayer = args.layer
        self.kernel = args.kernel
        self.sigma_val = args.sigma
        self.proj_inference_de = nn.Linear(self.bert_hidden_dim * 2, self.num_labels)
        self.proj_att = nn.Linear(self.kernel, 1)
        self.proj_input_de = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)

        self.proj_select = nn.Linear(self.kernel, 1)
        self.mu = Variable(torch.FloatTensor(kernal_mus(self.kernel)), requires_grad=False).view(1, 1, 1,
                                                                                                 self.kernel).to(
            self.device)
        self.sigma = Variable(torch.FloatTensor(kernel_sigmas(self.kernel, self.sigma_val)), requires_grad=False).view(
            1, 1, 1,
            self.kernel).to(self.device)
        self.tokenizer = tokenizer

        # ------------------------------------------
        # FND
        # ------------------------------------------

        config_kgat = config["KGAT"]
        type = torch.float32

        # self.alpha = nn.Parameter(torch.tensor([0.9]), requires_grad=True)  # Smoothing constant to avoid zero division
        self.pr_weight = torch.empty((3, args.evi_num), dtype=type)
        nn.init.uniform_(self.pr_weight)
        self.pr_param = nn.Parameter(self.pr_weight)
        self.mode = args.mode

        self.trans_mat_weight = torch.empty((self.evi_num, self.evi_num), device=self.device, dtype=type)

        # Translation matrix
        mean = config_kgat.getfloat("translation_mat_weight_mean")
        std = config_kgat.getfloat("translation_mat_weight_std")
        nn.init.normal_(self.trans_mat_weight, mean, std)
        self.param_trans_mat = nn.Parameter(self.trans_mat_weight, requires_grad=True)

        config_s = config["gat.social"]
        n_channels = config_s.getint("num_tweets_in_each_pair", 6)

        self.proj_pred_P = nn.Linear(n_channels, 1)

        self.att_prior_P = nn.Linear(self.args.num_tweets, 1, bias=False)
        self.att_prior_U = nn.Linear(self.args.num_users, 1, bias=False)
        self.att_prior_K = nn.Linear(self.args.num_words_per_topic, 1, bias=False)
        nn.init.normal_(self.att_prior_P.weight, mean, std)
        nn.init.normal_(self.att_prior_U.weight, mean, std)
        nn.init.normal_(self.att_prior_K.weight, mean, std)

        mean = config_kgat.getfloat("linear_weight_mean")
        std = config_kgat.getfloat("linear_weight_std")

        self.proj_pred_interact = nn.Linear(2, 1)
        self.param_pred_K = torch.empty(1)
        nn.init.normal_(self.param_pred_K, mean, std)

        # ------------------------------------------
        # For User features
        # ------------------------------------------

        self.user_embed_dim = args.user_embed_dim
        self.num_users = args.num_users

        self.user_emb = Linear(len(fields_user), self.user_embed_dim, bias=False)
        nn.init.normal_(self.user_emb.weight, mean, std)

        if self.mode == "FF-concat":
            """
            This part has been changed for concatenating
            Ignore this part for now
            """
            self.proj_gat = nn.Sequential(
                Linear(self.bert_hidden_dim * 2, 128),
                ReLU(True),
                Linear(128, 1)
            )
            self.proj_gat_usr = nn.Sequential(
                Linear(self.bert_hidden_dim * 2 + self.user_embed_dim, 128, bias=False),
                ReLU(True),
                Linear(128, 1, bias=False)
            )


        else:
            self.proj_gat = nn.Sequential(
                Linear(self.bert_hidden_dim * 2, 128),
                ReLU(True),
                Linear(128, 1)
            )

            self.proj_gat_usr = nn.Sequential(
                Linear(self.user_embed_dim, 128, bias=False),
                ReLU(True),
                Linear(128, 1, bias=False)
            )
            self.proj_user = Linear(self.user_embed_dim * 2, self.bert_hidden_dim * 2, bias=False)
            nn.init.normal_(self.proj_user.weight, mean, std)
        nn.init.normal_(self.proj_gat_usr[0].weight, mean, std)
        nn.init.normal_(self.proj_gat_usr[2].weight, mean, std)

        self.add_user_embed()


    def add_user_embed(self):
        all_user_embed_d, _ = load_user_embed_and_Rs(self.args, get_root_dir())
        self.user_embeds_indices_d = {}

        # Num_examples, 5, 32, 64
        user_embeds_tensor = torch.zeros(
            size=(len(all_user_embed_d), self.evi_num, self.args.num_users, self.args.user_embed_dim))
        for i, filename in enumerate(all_user_embed_d):  # .items():
            user_embeds_tensor[i] = all_user_embed_d[filename]
            self.user_embeds_indices_d[filename] = i

        user_embeds_tensor = user_embeds_tensor.reshape(-1, self.args.user_embed_dim)

        self.user_embeds = nn.Embedding.from_pretrained(user_embeds_tensor).to(self.device)

    def load_user_embed(self, step, user_metadata):
        if self.args.pretrained_user_embed:
            user_num_batch = self.evi_num * self.args.num_users * self.batch_size
            indices = torch.arange(step * user_num_batch, (step + 1) * user_num_batch, 1).to(self.device)
            user_embed = self.user_embeds(indices).to(self.device)
            user_embed = user_embed.reshape(self.batch_size, self.evi_num, self.args.num_users, self.args.user_embed_dim)
        else:
            user_embed = self.user_emb(user_metadata).to(self.device)

        return user_embed

    def att_prior_mr(self, R_p, R_u, R_k):
        H_p = self.att_prior_P(R_p)  # (B, 5, 6)
        H_u = self.att_prior_U(R_u)  # (B, 5, 32)
        H_k = self.att_prior_K(R_k)  # (B, 5, 7)
        delta = H_p + H_u + H_k
        return delta

    def self_attention_usr(self, inputs, inputs_hiddens, mask, index, z_qv_z_v_all=None):
        """
        Models interactions among user embeddings
        """

        idx = torch.LongTensor([index]).to(self.device)
        mask = mask.view([-1, self.evi_num, self.num_users]).to(self.device)

        # Hidden feature of ONLY current node
        # B, num_user, user_embed_dim
        own_hidden = torch.index_select(inputs_hiddens, 1, idx).to(self.device)
        # B, 1, num_user
        own_mask = torch.index_select(mask, 1, idx)

        # B, 1, user_embed_dim
        # x^v of current (one) user
        own_input = torch.index_select(inputs, 1, idx).to(self.device)

        # B, 5, user_embed_dim
        own_input = own_input.repeat(1, self.evi_num, 1)

        # Hidden feature of ONLY current node
        # B, 5, num_user, user_embed_dim
        own_hidden = own_hidden.repeat(1, self.evi_num, 1, 1)

        # B, 5, num_user
        own_mask = own_mask.repeat(1, self.evi_num, 1)

        # B*5, num_user, user_embed_dim
        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)

        # B*5, num_user, user_embed_dim
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        # B*5, num_user
        # gamma: Importance of each user

        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, self.num_users, self.user_embed_dim),
                                                  own_norm.view(-1, self.num_users, self.user_embed_dim),
                                                  own_mask.view(-1, self.num_users), own_mask.view(-1, self.num_users))

        # B, 5, num_user
        # gamma: Importance of each user
        att_score = att_score.view(-1, self.evi_num, self.num_users, 1)

        # B, 5, user_embed_dim
        # Token-wise weighted average
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)

        if self.mode == "FF-concat":
            z_qv_z_v = z_qv_z_v_all[:, index, :, :]
            concat_att_embed = torch.cat([z_qv_z_v, denoise_inputs], dim=1)
        else:
            concat_att_embed = denoise_inputs

        # weight_de = torch.cat([own_input, denoise_inputs], -1)
        weight_de = self.proj_gat_usr(concat_att_embed)
        weight_de = F.softmax(weight_de, dim=1)
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs_de

    def self_attention(self, inputs, inputs_hiddens, mask, mask_evidence, index, trans_mat_prior=None):
        idx = torch.LongTensor([index]).to(self.device)

        # inputs: B, 5, 768
        # inputs_hiddens: B, 5, 130, 768

        # B, 5, 130
        mask = mask.view([-1, self.evi_num, self.max_len])
        # B, 5, 130
        mask_evidence = mask_evidence.view([-1, self.evi_num, self.max_len])
        # B, 130, 768
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)

        # B, 1, 130
        own_mask = torch.index_select(mask, 1, idx)

        # B, 1, 768
        # z^v of current (one) claim-evi pair
        own_input = torch.index_select(inputs, 1, idx)
        # B, 5, 130, 768
        own_hidden = own_hidden.repeat(1, self.evi_num, 1, 1)

        # B, 5, 130
        own_mask = own_mask.repeat(1, self.evi_num, 1)

        # B, 5, 768
        own_input = own_input.repeat(1, self.evi_num, 1)

        # B*5, 130
        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)

        # B, 5, 130, 768
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        # B*5, 130
        # alpha: Importance of each token
        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, self.max_len, self.bert_hidden_dim),
                                                  own_norm.view(-1, self.max_len, self.bert_hidden_dim),
                                                  mask_evidence.view(-1, self.max_len), own_mask.view(-1, self.max_len))

        # B, 5, 130, 1
        # alpha: Token-wise weighted average
        att_score = att_score.view(-1, self.evi_num, self.max_len, 1)

        # B, 5, 768

        # z^{q -> p}
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)

        # B, 5, 1536
        # z^q || z^{p}
        weight_inp = torch.cat([own_input, inputs], -1)

        # B, 5, 1536
        z_q_z_v = weight_inp

        # MLP()
        # B, 5, 1
        weight_inp = self.proj_gat(weight_inp)

        # gamma
        # B, 5, 1
        weight_inp = F.softmax(weight_inp, dim=1)

        outputs = (inputs * weight_inp).sum(dim=1)

        # B, 5, 1536
        # z^p || z^{q -> p}
        # Can be changed into z^p || z^{q}
        if self.mode == "FF-P":
            weight_de = z_q_z_v  # shallow copy
            denoise_inputs = inputs.clone()
        else:
            weight_de = torch.cat([own_input, denoise_inputs], -1)

        z_qv_z_v = weight_de

        # gamma
        weight_de = self.proj_gat(weight_de)
        if trans_mat_prior is not None:
            weight_de = torch.cat([weight_de, trans_mat_prior[:, index].reshape(-1, self.evi_num, 1)], dim=2)
            weight_de = self.proj_pred_interact(weight_de)
        weight_de = F.softmax(weight_de, dim=1)

        # \sum {gamma^{q->p} * \hat{z}^{q->p}}
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de, z_qv_z_v

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)

        # 5, 130, 130, B
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1],
                                                                      d_embed.size()[1], 1)
        # 5, 130, 130, 11
        pooling_value = torch.exp(
            (- ((sim - self.mu.to(self.device)) ** 2) / (self.sigma.to(self.device) ** 2) / 2)) * attn_d
        # 5, 130, 11
        pooling_sum = torch.sum(pooling_value, 2)  # If merge content and social representation here
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q

        # 5, 130, 11
        log_pooling_sum_all = torch.sum(log_pooling_sum, 1) / (torch.sum(attn_q, 1) + 1e-10)

        # 5, 130, 1
        log_pooling_sum = self.proj_select(log_pooling_sum_all).view([-1, 1])
        return log_pooling_sum, log_pooling_sum_all

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1],
                                                                      d_embed.size()[1], 1)

        # B, 130, 11
        pooling_value = torch.exp(
            (- ((sim - self.mu.to(self.device)) ** 2) / (self.sigma.to(self.device) ** 2) / 2)) * attn_d

        # B*5, num_users, 11
        log_pooling_sum = torch.sum(pooling_value, 2)

        # B*5, num_users, 11
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))

        # B*5, num_users, 1
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)

        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(), -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum

    def unpack_inputs(self, inputs):
        inp_tensor, msk_tensor, seg_tensor, step = inputs
        if self.use_cuda:
            msk_tensor = msk_tensor.view(-1, self.max_len).cuda()
            inp_tensor = inp_tensor.view(-1, self.max_len).cuda()
            seg_tensor = seg_tensor.view(-1, self.max_len).cuda()
        else:
            msk_tensor = msk_tensor.view(-1, self.max_len)
            inp_tensor = inp_tensor.view(-1, self.max_len)
            seg_tensor = seg_tensor.view(-1, self.max_len)
        return inp_tensor, msk_tensor, seg_tensor, step

    def predict_prior(self, score):
        prior = self.proj_score(score)
        return prior

    def reshape_input_and_masks(self, inputs_hiddens, msk_tensor, seg_tensor):

        # B*5, 130
        mask_text = msk_tensor.view(-1, self.max_len).float()

        # First token ([CLS]) set to 0
        # B*5, 1
        mask_text[:, 0] = 0.0

        # Claim part set to 1 (Except first token [CLS])
        mask_claim = (1 - seg_tensor.float()) * mask_text

        # Evidence part set to 1
        mask_evidence = seg_tensor.float() * mask_text

        # z^p or h_p^0, Hidden representation of first token
        inputs_hiddens = inputs_hiddens.view(-1, self.max_len, self.bert_hidden_dim)

        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)
        return mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm

    def channel_text(self, inputs_hiddens, inputs, msk_tensor, seg_tensor, delta):
        mask_text, mask_claim, mask_evidence, inputs_hiddens, inputs_hiddens_norm = self.reshape_input_and_masks(
            inputs_hiddens, msk_tensor, seg_tensor)

        # ------------------------------------------
        # Evidence Selection: P(K^v, G)
        # ------------------------------------------
        # B*5, 1
        # log_pooling_sum_all: Content signals at several kernels
        log_pooling_sum, log_pooling_sum_all = self.get_intersect_matrix(inputs_hiddens_norm, inputs_hiddens_norm,
                                                                         mask_claim,
                                                                         mask_evidence)

        # B, 5, 1
        log_pooling_sum = log_pooling_sum.view([-1, self.evi_num, 1])

        if not self.args.mode == "FF-I":
            log_pooling_sum += delta

        # P(K^v, G)
        # B, 5, 1
        select_prob = F.softmax(log_pooling_sum, dim=1)

        # ------------------------------------------
        # Claim Label Prediction: P(y|K^p, G)
        # ------------------------------------------
        # B, 5, 768
        inputs = inputs.view([-1, self.evi_num, self.bert_hidden_dim])

        # B, 5, 130, 768
        inputs_hiddens = inputs_hiddens.view([-1, self.evi_num, self.max_len, self.bert_hidden_dim])

        inputs_att_de = []
        z_qv_z_v_all = []

        for i in range(self.evi_num):
            # outputs_de: z^{q->v} B, 768
            outputs, outputs_de, z_qv_z_v = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
            z_qv_z_v_all.append(z_qv_z_v)

        # B, 5, 768
        # All z^{v}, same as `inputs`
        inputs_att = inputs.view([-1, self.evi_num, self.bert_hidden_dim])

        # hstack
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        z_qv_z_v_all = torch.cat(z_qv_z_v_all, dim=1)

        # B, 5, 768
        # All z^{q->v}
        inputs_att_de = inputs_att_de.view([-1, self.evi_num, self.bert_hidden_dim])
        z_qv_z_v_all = z_qv_z_v_all.view([-1, self.evi_num, self.evi_num, self.bert_hidden_dim])

        return select_prob, inputs_att, inputs_att_de, z_qv_z_v_all

    def channel_usr(self, user_hiddens, z_qv_z_v_all=None):

        # ------------------------------------------
        # Evidence Selection: P(\hat{K}^p, G)
        # ------------------------------------------

        user_hiddens = user_hiddens.reshape(-1, self.evi_num, self.num_users, self.user_embed_dim)

        mask_usr = torch.ones_like(user_hiddens[:, :, :, 0])

        inputs_att_de_usr = []

        user_inputs = torch.mean(user_hiddens, dim=2)
        # user_inputs = torch.max(user_hiddens, dim=2)[0]

        for i in range(self.evi_num):
            outputs_de_usr = self.self_attention_usr(user_inputs, user_hiddens, mask_usr, i, z_qv_z_v_all)
            inputs_att_de_usr.append(outputs_de_usr)

        # B, 5, 768
        # All x^{v}, same as `inputs`
        inputs_att_usr = user_inputs.view([-1, self.evi_num, self.user_embed_dim])

        # hstack
        inputs_att_de_usr = torch.cat(inputs_att_de_usr, dim=1)

        inputs_att_de_usr = inputs_att_de_usr.view([-1, self.evi_num, self.user_embed_dim])

        return inputs_att_usr, inputs_att_de_usr

    def forward(self, inputs, Rs, user_metadata):
        inp_tensor, msk_tensor, seg_tensor, step = self.unpack_inputs(inputs)

        # msk_tensor: (B * evi_num, 130)
        # seg_tensor: (B * evi_num, 130)
        # mask_text: Every sentence starts with 0, followed by
        # 1's at positions where tokens are present

        inputs_hiddens, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)

        # Attention prior
        delta = None
        if Rs is not None:
            R_p, R_u, R_k = Rs
            delta = self.att_prior_mr(R_p, R_u, R_k)

        select_prob, inputs_att, inputs_att_de, z_qv_z_v_all = self.channel_text(inputs_hiddens, inputs, msk_tensor,
                                                                                 seg_tensor, delta)

        # All z^{q->v} || All z^{v}
        # B, 5, 768 || B, 5, 768 -> B, 5, 1536
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        user_embed = self.load_user_embed(step, user_metadata)

        if self.mode in ["FF"]:
            # Initialize our embedding module from the embedding matrix
            inputs_att_usr, inputs_att_de_usr = self.channel_usr(user_embed, z_qv_z_v_all)

            # All z^{q->p} || All x^{v}
            # B, 5, user_emb_dim || B, 5, user_emb_dim -> B, 5, user_emb_dim*2
            inputs_att_usr_combined = torch.cat([inputs_att_usr, inputs_att_de_usr], -1)

        # TODO: If use user features, add a fusion gate here

        if self.mode in ["FF"]:
            inputs_att_usr_combined = self.proj_user(inputs_att_usr_combined)

            # TODO: project user features
            assert inputs_att_usr_combined.shape == inputs_att.shape

            inputs_att = inputs_att + inputs_att_usr_combined

        # v
        # B, 5, 2
        inference_feature = self.proj_inference_de(inputs_att)

        # B, 5, 2
        class_prob = F.softmax(inference_feature, dim=2)

        if self.mode == "FF-I":
            select_prob = torch.ones_like(select_prob) / self.evi_num

        prob = torch.sum(select_prob * class_prob, 1)

        logits = torch.log(prob)

        return logits, prob
