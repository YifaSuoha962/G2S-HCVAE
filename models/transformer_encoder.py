import torch
import torch.nn as nn
from typing import Dict, List

from .module_utils import RelTransformerBase, RelMultiHeadAttention, GLU, RMSNorm, PosWiseFFN
from .dist import Normal


class RelTransformer_Encoder_Block(nn.Module):
    def __init__(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        dropout=0.1,
        block_id=0,
        alpha=None,
        init_beta=None,
        dist_range=[-1, -1],
        **kwargs
    ):
        super(RelTransformer_Encoder_Block, self).__init__(**kwargs)
        self.block_id = block_id
        self.alpha = alpha
        self.init_beta = init_beta
        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            u0=u0,
            v0=v0,
            u1=u1,
            v1=v1,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict0=rel_dict0,
            rel_dict1=rel_dict1,
            dropout=dropout,
            dist_range=dist_range
        )

    def _build_model(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        dropout=0.1,
        dist_range=-1
    ):
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.self_attention = RelMultiHeadAttention(
            d_model=d_model,
            h=h,
            u=u0,
            v=v0,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict=rel_dict0,
            dropout=dropout,
            dist_range=dist_range
        )
        self.ffn = GLU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        if self.init_beta is not None:
            self.self_attention._init_param(vo_beta=self.init_beta)
            if self.ffn.__class__.__name__ == 'GLU':
                self.ffn._init_param(vo_beta=1.)
            elif self.ffn.__class__.__name__ == 'PosWiseFFN':
                self.ffn._init_param(vo_beta=self.init_beta)

    def forward(
        self,
        x: torch.Tensor,
        src_len: List[torch.Tensor],
        dist=None,  # for graph distance encoding
        deg=None,  # for graph deg encoding
        edge=None  # for graph edge bias
    ):
        x_in, _ = self.self_attention(
            q=x,
            k=x,
            v=x,
            length=src_len,
            dist=dist,
            deg=deg,
            edge=edge
        )
        x = self.dropout1(x_in) + x * self.alpha
        x = self.norm1(x)
        x = self.dropout2(self.ffn(x)) + x * self.alpha
        x = self.norm2(x)
        return x


class RelTransformer_Encoder(RelTransformerBase):
    def __init__(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        rel_apply='add',
        dropout=0.1,
        alpha=None,
        init_beta=None,
        dist_range=None,
        **kwargs
    ):
        super(RelTransformer_Encoder, self).__init__(
            d_rel0=d_model,
            d_rel1=d_model,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            rel_apply=rel_apply,
            **kwargs
        )
        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            layer=layer,
            pe=pe,
            sin_enc=sin_enc,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            dist_range=dist_range
        )

    def _build_model(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        dropout=0.1,
        alpha=None,
        init_beta=None,
        dist_range=None,
    ):
        if dist_range is None:
            dist_range = [-1 for _ in range(layer)]
        self.block = nn.Sequential()
        for i in range(layer):
            self.block.add_module(
                'block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,
                    rel_dict0=attn0_rel_dict,
                    rel_dict1=attn1_rel_dict,
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._attention_score = [None] * len(self.block)

    def forward(
        self,
        # size(batch, seq_len, d_model), need to be scaled before forward
        src: torch.Tensor,
        src_len: List[torch.Tensor],
        dist=None,  # for graph distance encoding
        deg=None,  # for graph deg encoding
        edge=None,  # for graph edge bias
    ):
        if edge is not None:
            for i, block in enumerate(self.block):
                src = block(
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                )
        else:
            for i, block in enumerate(self.block):
                src = block(
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )
        return src.contiguous()

    @property
    def _attention_weight(self) -> List:
        return self._attention_score


# Layer-Wised latent Trnasformer
class Two_pass_Lv_Transformer_Encoder(RelTransformerBase):
    def __init__(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            rel_apply='add',
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,
            kl_threshold=0,
            **kwargs
    ):
        super(Two_pass_Lv_Transformer_Encoder, self).__init__(
            d_rel0=d_model,
            d_rel1=d_model,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            rel_apply=rel_apply,
            **kwargs
        )

        self.layer = layer
        self.lat_sz = lat_sz
        self.kl_threshold = kl_threshold

        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,                    # head
            layer=layer,
            pe=pe,
            sin_enc=sin_enc,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            dist_range=dist_range,
            lat_sz=lat_sz,
            kl_threshold=kl_threshold
        )


    def _build_model(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,                      # 64 in default
            kl_threshold=0           # 0  in default
    ):
        if dist_range is None:
            dist_range = [-1 for _ in range(layer)]

        self.first_block = nn.Sequential()
        for i in range(layer):
            self.first_block.add_module(
                'first_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=None,
                    pe='none',                  # for product graph
                    rel_dict0=attn0_rel_dict,
                    rel_dict1=attn1_rel_dict,
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._first_attention_score = [None] * len(self.first_block)
        self.tanh_activate = nn.Tanh()

        self.second_block = nn.Sequential()
        for i in range(layer):
            self.second_block.add_module(
                'second_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,                  # for reactant sequence
                    rel_dict0={},
                    rel_dict1={},
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._second_attention_score = [None] * len(self.second_block)

        assert len(self.first_block) == len(self.second_block)

        # layer-wise cvae
        # TODO: 是不是得把 (x, y) 拼起来
        # self.recog_block = nn.ModuleList(
        #     [nn.Linear(h + lat_sz, 2 * lat_sz) for _ in range(layers)])
        self.recog_block = nn.ModuleList(
            [nn.Linear(d_model * 2 + self.lat_sz, 2 * self.lat_sz) for _ in range(self.layer)])
        self.prior_blcok = nn.ModuleList(
                [nn.Linear(self.lat_sz + d_model, 2 * self.lat_sz) for _ in range(self.layer)])
        # memorize latent variales?
        self.reccurnt_cell_weight_hh = nn.ModuleList(
            [nn.Linear(self.lat_sz, lat_sz) for _ in range(self.layer)])
        self.reccurnt_cell_weight_ih = nn.ModuleList(
            [nn.Linear(self.lat_sz, self.lat_sz) for _ in range(self.layer)])

        self.mu_bn = nn.BatchNorm1d(self.lat_sz)
        self.mu_bn.weight.requires_grad = True
        self.gamma = 0.5

    def _recurrent_latent(self, post_latent, prior_latent_representaion, weight_ih, weight_hh):
        # like rnn
        hidden = self.tanh_activate(weight_hh(prior_latent_representaion) + weight_ih(post_latent))
        return hidden

    def get_prior(self, src, src_len, dist, deg, edge):
        """
        :param condition: product graph [batch_sz, hidden_size]
        :return:
        """
        device = src.device
        batch_size = src.shape[0]
        # prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        prior_latent = torch.zeros_like(prior_latent_representaion)
        all_prior_latent = ()

        for i in range(self.layer):
            if edge is not None:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                )
            else:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )
            mean_src = src.mean(dim=1)
            prior_network = self.prior_blcok[i]
            weight_ih = self.reccurnt_cell_weight_ih[i]
            weight_hh = self.reccurnt_cell_weight_hh[i]
            prior_latent_representaion = self._recurrent_latent(prior_latent, prior_latent_representaion, weight_ih,
                                                                weight_hh)
            prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)
            # sample from p(z(l) | z(<l), x)
            prior_mu, prior_sigma = torch.chunk(prior_network(prior_representaion), 2, dim=-1)
            prior_dist = Normal(prior_mu, prior_sigma)
            prior_latent, _ = prior_dist.sample()
            all_prior_latent = all_prior_latent + (prior_latent,)

        return src, all_prior_latent, prior_latent_representaion

    def forward(
            self,
            # size(batch, seq_len, d_model), need to be scaled before forward
            src: torch.Tensor,
            src_len: List[torch.Tensor],
            tgt: torch.Tensor,
            tgt_len: List[torch.Tensor],
            dist=None,  # for graph distance encoding
            deg=None,  # for graph deg encoding
            edge=None,  # for graph edge bias
    ):
        # first_pass： only encode product graph
        kl_loss = 0

        # init distribution
        batch_size = src.shape[0]
        device = src.device
        prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        post_latent = torch.zeros_like(prior_latent_representaion)

        all_mean_src = ()
        all_mean_tgt = ()
        all_post_latent = ()        # sampled latent variables
        all_kl_loss = ()
        all_log_prior = ()
        all_log_post = ()
        all_post_mu = ()
        all_post_sigma = ()

        for i in range(self.layer):
            if edge is not None:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                    )
            else:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )
            mean_src = src.mean(dim=1)
            # all_mean_src = all_mean_src + (mean_src,)
            tgt = self.second_block[i](
                    x=tgt,
                    src_len=tgt_len,
                    dist=None,
                    deg=None,
                    edge=None
                )
            mean_tgt = tgt.mean(dim=1)
            # all_mean_tgt = all_mean_tgt + (mean_tgt,)

            prior_latent_representaion = self._recurrent_latent(post_latent, prior_latent_representaion,
                                                                self.reccurnt_cell_weight_ih[i],
                                                                self.reccurnt_cell_weight_hh[i],)
            prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)

            # p(z(l) | z(<l), x)
            prior_mu, prior_sigma = torch.chunk(self.prior_blcok[i](prior_representaion), 2, dim=-1)

            prior_dist = Normal(prior_mu, prior_sigma)
            # p(z(l) | z(<l), x, y)
            post_latent_representation = self.recog_block[i](
                torch.cat((prior_latent_representaion, mean_src, mean_tgt), dim=-1))
            post_mu, post_sigma = torch.chunk(post_latent_representation, 2, dim=-1)

            # sample from post dist : p(z(l) | z(<l), x, y)
            # batch norm on mu
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            post_mu = self.mu_bn(post_mu)

            post_dist = Normal(post_mu, post_sigma)
            post_latent, _ = post_dist.sample()

            # compute kl
            kl_loss = post_dist.kl(prior_dist)
            kl_threshold = torch.Tensor([self.kl_threshold]).type_as(kl_loss)
            kl_loss = torch.max(kl_loss, kl_threshold)

            log_prior_z = prior_dist.log_p(post_latent)
            log_post_z = post_dist.log_p(post_latent)

            all_post_latent = all_post_latent + (post_latent,)
            all_kl_loss = all_kl_loss + (kl_loss,)
            all_log_prior = all_log_prior + (log_prior_z,)
            all_log_post = all_log_post + (log_post_z,)
            all_post_mu = all_post_mu + (post_mu,)
            all_post_sigma = all_post_sigma + (post_sigma,)

            kl_loss = torch.stack(all_kl_loss).mean(0)
            log_prior = torch.stack(all_log_prior).mean(0)
            log_post = torch.stack(all_log_post).mean(0)

            mu = torch.cat(all_post_mu, dim=-1)
            sigma = torch.cat(all_post_sigma, dim=-1)

        # 第一趟只要 src: node_feature of product
        # 第二趟只要 kl_loss 和 latent_st
        return src.contiguous(), kl_loss, all_post_latent, mu, sigma, log_prior, log_post, prior_latent_representaion


    @property
    def _attention_weight(self) -> List:
        return self._first_attention_score, self._second_attention_score


class Dual_way_Lv_Transformer_Encoder(RelTransformerBase):
    def __init__(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            rel_apply='add',
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,
            kl_threshold=0,
            cvae_layers=1,
            **kwargs
    ):
        super(Dual_way_Lv_Transformer_Encoder, self).__init__(
            d_rel0=d_model,
            d_rel1=d_model,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            rel_apply=rel_apply,
            **kwargs
        )

        self.layer = layer
        self.lat_sz = lat_sz
        self.kl_threshold = kl_threshold

        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,                    # head
            layer=layer,
            pe=pe,
            sin_enc=sin_enc,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            dist_range=dist_range,
            lat_sz=lat_sz,
            kl_threshold=kl_threshold,
            cvae_layers=cvae_layers
        )


    def _build_model(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,
            kl_threshold=0,
            cvae_layers=1
    ):
        if dist_range is None:
            dist_range = [-1 for _ in range(layer)]

        self.first_block = nn.Sequential()
        for i in range(layer):
            self.first_block.add_module(
                'first_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=None,
                    pe='none',                  # for product graph
                    rel_dict0=attn0_rel_dict,
                    rel_dict1=attn1_rel_dict,
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._first_attention_score = [None] * len(self.first_block)
        self.tanh_activate = nn.Tanh()

        self.second_block = nn.Sequential()
        for i in range(layer):
            self.second_block.add_module(
                'second_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,                  # for reactant sequence
                    rel_dict0={},
                    rel_dict1={},
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._second_attention_score = [None] * len(self.second_block)

        assert len(self.first_block) == len(self.second_block)

        # layer-wise cvae

        self.cvae_layers = cvae_layers
        self.cvae_thresh = self.layer - self.cvae_layers
        # TODO: 是不是得把 (x, y) 拼起来
        self.recog_block = nn.ModuleList(
            [nn.Linear(d_model * 2 + self.lat_sz, 2 * self.lat_sz) for _ in range(self.cvae_layers)])
        self.prior_blcok = nn.ModuleList(
            [nn.Linear(self.lat_sz + d_model, 2 * self.lat_sz) for _ in range(self.cvae_layers)])
        # memorize latent variales?
        self.reccurnt_cell_weight_hh = nn.ModuleList(
            [nn.Linear(self.lat_sz, lat_sz) for _ in range(self.cvae_layers)])
        self.reccurnt_cell_weight_ih = nn.ModuleList(
            [nn.Linear(self.lat_sz, self.lat_sz) for _ in range(self.cvae_layers)])


        self.mu_bn = nn.BatchNorm1d(self.lat_sz)
        self.mu_bn.weight.requires_grad = True
        self.gamma = 0.5

    def _recurrent_latent(self, post_latent, prior_latent_representaion, weight_ih, weight_hh):
        # like rnn
        hidden = self.tanh_activate(weight_hh(prior_latent_representaion) + weight_ih(post_latent))
        return hidden

    def get_prior(self, src, src_len, dist, deg, edge):
        """
        :param condition: product graph [batch_sz, hidden_size]
        :return:
        """
        device = src.device
        batch_size = src.shape[0]
        # prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        prior_latent = torch.zeros_like(prior_latent_representaion)
        all_prior_latent = ()

        # 只有最后 i 层用 layer-wise lv
        for i in range(self.layer):
            if edge is not None:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                )
            else:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )

            left_layers = self.layer - i
            if left_layers <= self.cvae_layers:
                mean_src = src.mean(dim=1)

                prior_network = self.prior_blcok[self.cvae_layers - left_layers]
                weight_ih = self.reccurnt_cell_weight_ih[i - self.cvae_thresh]      # self.cvae_layers - left_layers
                weight_hh = self.reccurnt_cell_weight_hh[i - self.cvae_thresh]
                prior_latent_representaion = self._recurrent_latent(prior_latent, prior_latent_representaion, weight_ih,
                                                                    weight_hh)
                prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)
                # sample from p(z(l) | z(<l), x)
                prior_mu, prior_sigma = torch.chunk(prior_network(prior_representaion), 2, dim=-1)
                prior_dist = Normal(prior_mu, prior_sigma)
                prior_latent, _ = prior_dist.sample()
                all_prior_latent = all_prior_latent + (prior_latent,)

        return src, all_prior_latent

    def forward(
            self,
            # size(batch, seq_len, d_model), need to be scaled before forward
            src: torch.Tensor,
            src_len: List[torch.Tensor],
            tgt: torch.Tensor,
            tgt_len: List[torch.Tensor],
            dist=None,  # for graph distance encoding
            deg=None,  # for graph deg encoding
            edge=None,  # for graph edge bias
    ):
        # first_pass： only encode product graph
        kl_loss = 0

        # init distribution
        batch_size = src.shape[0]
        device = src.device
        prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        post_latent = torch.zeros_like(prior_latent_representaion)

        all_mean_src = ()
        all_mean_tgt = ()
        all_post_latent = ()        # sampled latent variables
        all_kl_loss = ()
        all_log_prior = ()
        all_log_post = ()
        all_post_mu = ()
        all_post_sigma = ()

        for i in range(self.layer):
            if edge is not None:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                    )
            else:
                src = self.first_block[i](
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )

            # all_mean_src = all_mean_src + (mean_src,)
            tgt = self.second_block[i](
                    x=tgt,
                    src_len=tgt_len,
                    dist=None,
                    deg=None,
                    edge=None
                )

            left_layers = self.layer - i

            if left_layers <= self.cvae_layers:
                mean_src = src.mean(dim=1)
                mean_tgt = tgt.mean(dim=1)

                prior_latent_representaion = self._recurrent_latent(post_latent, prior_latent_representaion,
                                                                    self.reccurnt_cell_weight_ih[i - self.cvae_thresh],
                                                                    self.reccurnt_cell_weight_hh[i - self.cvae_thresh], )
                prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)

                # p(z(l) | z(<l), x)
                prior_mu, prior_sigma = torch.chunk(self.prior_blcok[self.cvae_layers - left_layers](prior_representaion), 2, dim=-1)

                prior_dist = Normal(prior_mu, prior_sigma)
                # p(z(l) | z(<l), x, y)
                post_latent_representation = self.recog_block[i - self.cvae_thresh](
                    torch.cat((prior_latent_representaion, mean_src, mean_tgt), dim=-1))
                post_mu, post_sigma = torch.chunk(post_latent_representation, 2, dim=-1)

                # sample from post dist : p(z(l) | z(<l), x, y)
                # batch norm on mu
                ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
                self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
                post_mu = self.mu_bn(post_mu)

                post_dist = Normal(post_mu, post_sigma)
                post_latent, _ = post_dist.sample()

                # compute kl
                kl_loss = post_dist.kl(prior_dist)
                kl_threshold = torch.Tensor([self.kl_threshold]).type_as(kl_loss)
                kl_loss = torch.max(kl_loss, kl_threshold)

                log_prior_z = prior_dist.log_p(post_latent)
                log_post_z = post_dist.log_p(post_latent)

                all_post_latent = all_post_latent + (post_latent,)
                all_kl_loss = all_kl_loss + (kl_loss,)
                all_log_prior = all_log_prior + (log_prior_z,)
                all_log_post = all_log_post + (log_post_z,)
                all_post_mu = all_post_mu + (post_mu,)
                all_post_sigma = all_post_sigma + (post_sigma,)

                kl_loss = torch.stack(all_kl_loss).mean(0)
                log_prior = torch.stack(all_log_prior).mean(0)
                log_post = torch.stack(all_log_post).mean(0)

                mu = torch.cat(all_post_mu, dim=-1)
                sigma = torch.cat(all_post_sigma, dim=-1)

        # 第一趟只要 src: node_feature of product
        # 第二趟只要 kl_loss 和 latent_st
        return src.contiguous(), kl_loss, all_post_latent, mu, sigma, log_prior, log_post


    @property
    def _attention_weight(self) -> List:
        return self._first_attention_score, self._second_attention_score



class Two_pass_Lv_Seq_Transformer_Encoder(RelTransformerBase):
    def __init__(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            rel_apply='add',
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,
            kl_threshold=0,
            **kwargs
    ):
        super(Two_pass_Lv_Seq_Transformer_Encoder, self).__init__(
            d_rel0=d_model,
            d_rel1=d_model,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            rel_apply=rel_apply,
            **kwargs
        )

        self.layer = layer
        self.lat_sz = lat_sz
        self.kl_threshold = kl_threshold

        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,                    # head
            layer=layer,
            pe=pe,
            sin_enc=sin_enc,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            dist_range=dist_range,
            lat_sz=lat_sz,
            kl_threshold=kl_threshold
        )


    def _build_model(
            self, d_model, d_ff, h, layer,
            pe='abs',
            sin_enc=None,
            attn0_rel_dict={},
            attn1_rel_dict={},
            dropout=0.1,
            alpha=None,
            init_beta=None,
            dist_range=None,
            lat_sz=64,                      # 64 in default
            kl_threshold=0           # 0  in default
    ):
        if dist_range is None:
            dist_range = [-1 for _ in range(layer)]

        self.first_block = nn.Sequential()
        for i in range(layer):
            self.first_block.add_module(
                'first_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,  # for reactant sequence
                    rel_dict0={},
                    rel_dict1={},
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._first_attention_score = [None] * len(self.first_block)
        self.tanh_activate = nn.Tanh()

        self.second_block = nn.Sequential()
        for i in range(layer):
            self.second_block.add_module(
                'second_block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,                  # for reactant sequence
                    rel_dict0={},
                    rel_dict1={},
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._second_attention_score = [None] * len(self.second_block)

        assert len(self.first_block) == len(self.second_block)

        # layer-wise cvae
        # TODO: 是不是得把 (x, y) 拼起来
        # self.recog_block = nn.ModuleList(
        #     [nn.Linear(h + lat_sz, 2 * lat_sz) for _ in range(layers)])
        self.recog_block = nn.ModuleList(
            [nn.Linear(d_model * 2 + self.lat_sz, 2 * self.lat_sz) for _ in range(self.layer)])
        self.prior_blcok = nn.ModuleList(
                [nn.Linear(self.lat_sz + d_model, 2 * self.lat_sz) for _ in range(self.layer)])
        # memorize latent variales?
        self.reccurnt_cell_weight_hh = nn.ModuleList(
            [nn.Linear(self.lat_sz, lat_sz) for _ in range(self.layer)])
        self.reccurnt_cell_weight_ih = nn.ModuleList(
            [nn.Linear(self.lat_sz, self.lat_sz) for _ in range(self.layer)])

        self.mu_bn = nn.BatchNorm1d(self.lat_sz)
        self.mu_bn.weight.requires_grad = True
        self.gamma = 0.5

    def _recurrent_latent(self, post_latent, prior_latent_representaion, weight_ih, weight_hh):
        # like rnn
        hidden = self.tanh_activate(weight_hh(prior_latent_representaion) + weight_ih(post_latent))
        return hidden

    def get_prior(self, src, src_len, dist, deg, edge):
        """
        :param condition: product graph [batch_sz, hidden_size]
        :return:
        """
        device = src.device
        batch_size = src.shape[0]
        # prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        prior_latent = torch.zeros_like(prior_latent_representaion)
        all_prior_latent = ()

        for i in range(self.layer):

            src = self.first_block[i](
                x=src,
                src_len=src_len,
                dist=None,
                deg=None,
                edge=None
            )
            mean_src = src.mean(dim=1)
            prior_network = self.prior_blcok[i]
            weight_ih = self.reccurnt_cell_weight_ih[i]
            weight_hh = self.reccurnt_cell_weight_hh[i]
            prior_latent_representaion = self._recurrent_latent(prior_latent, prior_latent_representaion, weight_ih,
                                                                weight_hh)
            prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)
            # sample from p(z(l) | z(<l), x)
            prior_mu, prior_sigma = torch.chunk(prior_network(prior_representaion), 2, dim=-1)
            prior_dist = Normal(prior_mu, prior_sigma)
            prior_latent, _ = prior_dist.sample()
            all_prior_latent = all_prior_latent + (prior_latent,)

        return src, all_prior_latent

    def forward(
            self,
            # size(batch, seq_len, d_model), need to be scaled before forward
            src: torch.Tensor,
            src_len: List[torch.Tensor],
            tgt: torch.Tensor,
            tgt_len: List[torch.Tensor],
            dist=None,  # for graph distance encoding
            deg=None,  # for graph deg encoding
            edge=None,  # for graph edge bias
    ):
        # first_pass： only encode product graph
        kl_loss = 0

        # init distribution
        batch_size = src.shape[0]
        device = src.device
        prior_dist = Normal.get_standard(batch_size, self.lat_sz, device)
        prior_latent_representaion = torch.zeros(batch_size, self.lat_sz).to(device)
        post_latent = torch.zeros_like(prior_latent_representaion)

        all_mean_src = ()
        all_mean_tgt = ()
        all_post_latent = ()        # sampled latent variables
        all_kl_loss = ()
        all_log_prior = ()
        all_log_post = ()
        all_post_mu = ()
        all_post_sigma = ()

        for i in range(self.layer):
            src = self.first_block[i](
                x=src,
                src_len=src_len,
                dist=None,
                deg=None,
                edge=None
            )
            mean_src = src.mean(dim=1)
            # all_mean_src = all_mean_src + (mean_src,)
            tgt = self.second_block[i](
                    x=tgt,
                    src_len=tgt_len,
                    dist=None,
                    deg=None,
                    edge=None
                )
            mean_tgt = tgt.mean(dim=1)
            # all_mean_tgt = all_mean_tgt + (mean_tgt,)

            prior_latent_representaion = self._recurrent_latent(post_latent, prior_latent_representaion,
                                                                self.reccurnt_cell_weight_ih[i],
                                                                self.reccurnt_cell_weight_hh[i],)
            prior_representaion = torch.cat((prior_latent_representaion, mean_src), dim=-1)

            # p(z(l) | z(<l), x)
            prior_mu, prior_sigma = torch.chunk(self.prior_blcok[i](prior_representaion), 2, dim=-1)

            prior_dist = Normal(prior_mu, prior_sigma)
            # p(z(l) | z(<l), x, y)
            post_latent_representation = self.recog_block[i](
                torch.cat((prior_latent_representaion, mean_src, mean_tgt), dim=-1))
            post_mu, post_sigma = torch.chunk(post_latent_representation, 2, dim=-1)

            # sample from post dist : p(z(l) | z(<l), x, y)
            # batch norm on mu
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            post_mu = self.mu_bn(post_mu)

            post_dist = Normal(post_mu, post_sigma)
            post_latent, _ = post_dist.sample()

            # compute kl
            kl_loss = post_dist.kl(prior_dist)
            kl_threshold = torch.Tensor([self.kl_threshold]).type_as(kl_loss)
            kl_loss = torch.max(kl_loss, kl_threshold)

            log_prior_z = prior_dist.log_p(post_latent)
            log_post_z = post_dist.log_p(post_latent)

            all_post_latent = all_post_latent + (post_latent,)
            all_kl_loss = all_kl_loss + (kl_loss,)
            all_log_prior = all_log_prior + (log_prior_z,)
            all_log_post = all_log_post + (log_post_z,)
            all_post_mu = all_post_mu + (post_mu,)
            all_post_sigma = all_post_sigma + (post_sigma,)

            kl_loss = torch.stack(all_kl_loss).mean(0)
            log_prior = torch.stack(all_log_prior).mean(0)
            log_post = torch.stack(all_log_post).mean(0)

            mu = torch.cat(all_post_mu, dim=-1)
            sigma = torch.cat(all_post_sigma, dim=-1)

        # 第一趟只要 src: node_feature of product
        # 第二趟只要 kl_loss 和 latent_st
        return src.contiguous(), kl_loss, all_post_latent, mu, sigma, log_prior, log_post


    @property
    def _attention_weight(self) -> List:
        return self._first_attention_score, self._second_attention_score
