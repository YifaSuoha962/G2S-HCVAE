import math
import torch
import torch.nn as nn
from typing import Dict, List, Iterator
import torch.nn.functional as F

from utils.chem_tools import MAX_DIST, MAX_DEG
from utils.data_loader import G2SBatchData, S2SBatchData, graph_to_seq, bond_to_seq

from .module_utils import (AbsPositionalEncoding, Atom_Embedding, RMSNorm,
                           embedding_init, Focal_loss, LabelSmoothing_MCELoss)
from .graph_encoder import DMPNN_Encoder
from .transformer_encoder import RelTransformer_Encoder
from .transformer_decoder import RelTransformer_Decoder

# inference codes
from .infer_strategies.hugging_face_infer import Beam_Generate as Huggingface_Beam
from .infer_strategies.hugging_face_infer import Lv_Beam_Generate as Lv_Huggingface_Beam

# from .infer_strategies.onmt_infer import Beam_Generate as OpenNMT_Beam
from .infer_strategies.onmt_infer import Lv_Beam_Generate as Lv_OpenNMT_Beam

# CVAE module
# from .cvae_utils import BD_HCVAE_mod, DCVAE
from cvae_utils import gumbel_softmax, tile
from .dist import Normal

import math


class DCVAE(nn.Module):
    def __init__(self, lv_disc_size, hidden_size, temp=1.0):
        super(DCVAE, self).__init__()
        # gru
        self.prior_net = nn.GRU(input_size=hidden_size,
                                  hidden_size=lv_disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)
        self.recog_net = nn.GRU(input_size=hidden_size,
                                  hidden_size=lv_disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        self.temp = temp

    def forward(self, src_emb, tgt_emb):
        # avg attention pool
        recog_emb = torch.cat([src_emb, tgt_emb], dim=1).to(src_emb.device)
        logits = self.prior_net(src_emb, None)[-1][0]
        prior = F.log_softmax(logits, dim=-1)

        logits = self.recog_net(recog_emb, None)[-1][0]
        recog = F.softmax(logits, dim=-1)

        kld_loss = F.kl_div(prior, recog, reduction='sum')
        c_sample = gumbel_softmax(logits, hard=True)

        return c_sample, kld_loss

    def get_prior(self, src_emb):
        logits = self.prior_net(src_emb, None)[-1][0]
        c_sample = gumbel_softmax(logits, hard=True)
        return c_sample


class HCVAE(nn.Module):
    def __init__(self, lv_disc_size, hidden_size, temp=1.0):
        super(HCVAE, self).__init__()

        # gru
        self.prior_net = nn.GRU(input_size=hidden_size,
                                  hidden_size=lv_disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)
        self.recog_net = nn.GRU(input_size=hidden_size,
                                  hidden_size=lv_disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        self.temp = temp

    def forward(self, src_emb, tgt_emb):
        # avg attention pool
        recog_emb = torch.cat([src_emb, tgt_emb], dim=1).to(src_emb.device)
        logits = self.prior_net(src_emb, None)[-1][0]
        prior = F.log_softmax(logits, dim=-1)

        logits = self.recog_net(recog_emb, None)[-1][0]
        recog = F.softmax(logits, dim=-1)

        kld_loss = F.kl_div(prior, recog, reduction='sum')
        c_sample = gumbel_softmax(logits, hard=True)

        return c_sample, kld_loss

    def get_prior(self, src_emb):
        logits = self.prior_net(src_emb, None)[-1][0]
        c_sample = gumbel_softmax(logits, hard=True)
        return c_sample


class Graph_Transformer_VAE():
    def __init__(
            self,
            f_vocab: int,
            f_atom: int,
            f_bond: int,
            token_idx: Dict[str, int],
            cls_len: int,
            args,
            token_freq=None,
            token_count=None
    ):
        self.model = Graph_Transformer_Builder(
            f_vocab=f_vocab,
            f_atom=f_atom,
            f_bond=f_bond,
            token_idx=token_idx,
            cls_len=cls_len,
            args=args,
            token_freq=token_freq,
            token_count=token_count
        )

        self.model = self.model.to(args.device)

    def model_train(
            self,
            data: G2SBatchData,
            epoch,
            tot_epochs
    ):
        self.model.train()
        return self.model(data, epoch, tot_epochs)

    def model_predict(
            self,
            data: G2SBatchData,
            args,
            beam_size=10
    ):
        self.model.eval()
        return self.model.gen_predict(data, args, beam_size=beam_size)

    def collect_embeddings(self, data: G2SBatchData):
        self.model.eval()
        return self.model.encode_pairs(data)


    @property
    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.model.parameters()


class Graph_Transformer_Builder(nn.Module):
    def __init__(
            self,
            f_vocab: int,
            f_atom: int,
            f_bond: int,
            token_idx: Dict[str, int],
            cls_len: int,
            args,
            token_freq=None,
            token_count=None,
    ):
        super(Graph_Transformer_Builder, self).__init__()
        self.f_vocab = f_vocab
        self.f_atom = f_atom
        self.f_bond = f_bond
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.enc_head = args.enc_head
        self.dec_head = args.dec_head
        self.graph_layer = args.graph_layer
        self.enc_layer = args.enc_layer
        self.dec_layer = args.dec_layer
        self.max_dist = len(args.graph_dist_block)
        self.token_idx = token_idx
        self.token_freq = token_freq
        self.token_count = token_count
        self.use_subs = args.use_subs
        self.use_reaction_type = args.use_reaction_type
        self.model_lat_rxn = args.model_type == 'BiG2S_HCVAE'     # actually model the latent reaction type space
        self.dropout = args.dropout
        self.device = args.device
        self.decoder_cls = args.decoder_cls
        self.cls_len = cls_len
        # DeepNorm alpha [encoder, decoder, lat_generate]
        self.alpha = [0.81 * (((self.enc_layer ** 4) * self.dec_layer) ** (1 / 16)),
                      (3 * self.dec_layer) ** (1 / 4)]
        # DeepNorm beta [encoder, decoder, lat_generate]
        self.init_beta = [0.87 * (((self.enc_layer ** 4) * self.dec_layer) ** (-1 / 16)),
                          (12 * self.dec_layer) ** (-1 / 4)]

        # params of CVAE
        # Ensure lat_disc_size and temp are present in args
        assert hasattr(args, 'lat_disc_size'), "args must have attribute 'lat_disc_size'"
        assert hasattr(args, 'temp'), "args must have attribute 'temp'"

        self.lat_disc_size = args.lat_disc_size
        self.temp = args.temp

        self.cvae_type = args.cvae_type

        self._build_model()
        self._build_criterion(args)

    def _build_model(self):

        self.gnn_norm = RMSNorm(self.d_model)
        self.enc_dropout = nn.Dropout(self.dropout)
        self.dec_dropout = nn.Dropout(self.dropout)

        self.atom_embedding = Atom_Embedding(reaction_type= True if self.use_reaction_type else False)
        self.seq_embedding = nn.Embedding(self.f_vocab, self.d_model,
                                          padding_idx=self.token_idx['<PAD>'])
        if self.use_subs and self.cls_len > 0:
            self.bi_embedding = nn.Embedding(2, self.d_model)
        if self.use_reaction_type and self.cls_len > 0:
            self.cls_embedding = nn.Embedding(10, self.d_model)

        self.bond_project = nn.Linear(self.d_model, self.enc_head * self.enc_layer)
        if self.use_subs:
            self.out_project = nn.Sequential()
            self.out_project.add_module('prod', nn.Linear(self.d_model, self.f_vocab))
            self.out_project.add_module('subs', nn.Linear(self.d_model, self.f_vocab))
        else:
            self.out_project = nn.Linear(self.d_model, self.f_vocab)
        self.pos_encoding = AbsPositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout
        )
        self.graph_encoder = DMPNN_Encoder(
            f_node=self.f_atom,
            f_bond=self.f_bond,
            d_model=self.d_model,
            layer=self.graph_layer,
            dropout=self.dropout
        )
        self.transformer_encoder = RelTransformer_Encoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.enc_head,
            layer=self.enc_layer,
            pe='none',
            sin_enc=None,
            attn0_rel_dict={'deg': [MAX_DEG + 1, self.enc_head],
                'dist': [self.max_dist + 1, self.enc_head, self.max_dist]},
            attn1_rel_dict={},
            rel_apply='add',
            dropout=self.dropout,
            alpha=self.alpha[0],
            init_beta=self.init_beta[0],
            dist_range=None
        )
        self.transformer_decoder = RelTransformer_Decoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.dec_head,
            layer=self.dec_layer,
            pe='rope',
            sin_enc=self.pos_encoding.pos,
            attn0_rel_dict={},
            attn1_rel_dict={},
            rel_apply='add',
            use_subs=self.use_subs,
            dropout=self.dropout,
            alpha=self.alpha[1],
            init_beta=self.init_beta[1],
            cls_len=self.cls_len
        )

        # self.react_transformer_encoder = RelTransformer_Encoder(
        #     d_model=self.d_model,
        #     d_ff=self.d_ff,
        #     h=self.enc_head,
        #     layer=self.enc_layer,
        #     pe='rope',
        #     sin_enc=self.pos_encoding.pos,
        #     attn0_rel_dict={},
        #     attn1_rel_dict={},
        #     rel_apply='add',
        #     dropout=self.dropout,
        #     alpha=self.alpha[0],
        #     init_beta=self.init_beta[0],
        #     dist_range=None
        # )

        # modeling latent rxn space
        # if self.cvae_type == 'hcvae':
        #     self.cvae = BD_HCVAE_mod(self.d_model, self.lat_disc_size, self.d_model, self.temp)
        if self.cvae_type == 'dcvae':
            # lv_disc_size, lv_z_size, hidden_size, temp=1.0
            self.cvae = DCVAE(self.lat_disc_size, self.d_model, self.temp)
        # cls embedding is used for latent rxn
        self.cls_embedding = nn.Embedding(self.lat_disc_size, self.d_model)
        if self.cls_len > 0:
            self.cls_embedding = embedding_init(self.cls_embedding)

        self.seq_embedding = embedding_init(self.seq_embedding)
        if self.use_subs and self.cls_len > 0:
            self.bi_embedding = embedding_init(self.bi_embedding)
        if self.use_reaction_type and self.cls_len > 0:
            self.cls_embedding = embedding_init(self.cls_embedding)

        nn.init.xavier_normal_(self.bond_project.weight)
        for param in self.out_project.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param)

    def _build_criterion(self, args):
        self.main_criterion = Focal_loss(
                vocab_dict=self.token_idx,
                ignore_index=self.token_idx['<PAD>'],
                ignore_min_count=args.ignore_min_count,
                label_smooth=args.label_smooth,
                max_scale_rate=args.max_scale_rate,
                gamma=args.gamma,
                margin=args.margin,
                sentence_scale=args.sentence_scale,
                device=self.device,
                loss_type=args.loss_type
        )
        if args.vae_warmup > 0:
            self.kl_weight = 0
        else:
            self.kl_weight = 1.0

    def encode(self, data: G2SBatchData):
        if self.use_reaction_type:
            reaction_type_mess = data.reaction_type.repeat_interleave(data.src_graph_len, dim=0).unsqueeze(-1) - 1
            node_emb = torch.cat([data.src_node, reaction_type_mess], dim=-1)
            node_emb = self.atom_embedding(node_emb)
        else:
            node_emb = self.atom_embedding(data.src_node)
        bond_emb = data.src_bond.to(torch.float)
        node_emb, bond_emb = self.graph_encoder(
            node=node_emb,
            connect=data.src_link,
            bond=bond_emb,
            bond_neighbour=data.src_bond_neighbour)
        node_emb = self.gnn_norm(node_emb)
        node_emb = self.enc_dropout(node_emb)
        node_emb, deg_emb = graph_to_seq(node_feat=node_emb, node_num=data.src_graph_len),\
            graph_to_seq(node_feat=data.src_deg, node_num=data.src_graph_len)[..., 0]
        bond_emb = bond_to_seq(
            bond=bond_emb,
            connect=data.src_link,
            node_num=data.src_graph_len,
            bond_batch_idx=data.src_bond_idx)

        bond_emb = self.bond_project(bond_emb).split(self.enc_head, dim=-1)

        enc_mem = self.transformer_encoder(
            src=node_emb,
            src_len=[data.src_graph_len],
            dist=data.src_dist,
            deg=deg_emb,
            edge=bond_emb)
        return enc_mem

    def encode_ref(self, data: G2SBatchData):
        ref_emb = self.seq_embedding(data.tgt_seq)
        enc_ref_mem = self.react_transformer_encoder(
            src=ref_emb,
            src_len=[data.tgt_seq_len],
            dist=None,
            deg=None,
            edge=None)
        return enc_ref_mem

    def gen_loss(
            self,
            dec_out: torch.Tensor,
            data: G2SBatchData,
            kld_loss: float,
            epoch: int,
            tot_epochs: int
    ):
        dec_out = dec_out.transpose(1, 2)
        sub_loss = None
        sub_loss_record = [0, 0, 0]

        if self.use_subs:
            p2s_idx, s2p_idx = data.bi_label.eq(0), data.bi_label.eq(1)
            p2s_tgt, s2p_tgt = data.tgt_seq[p2s_idx], data.tgt_seq[s2p_idx]
            p2s_dec, s2p_dec = dec_out[p2s_idx], dec_out[s2p_idx]
            p2s_loss, p2s_true_token, p2s_all_token = self.main_criterion(input=p2s_dec, target=p2s_tgt,
                                                                          seq_len=data.tgt_seq_len[p2s_idx], epoch=epoch, tot_epochs=tot_epochs)
            s2p_loss, s2p_true_token, s2p_all_token = self.main_criterion(input=s2p_dec, target=s2p_tgt,
                                                                          seq_len=data.tgt_seq_len[s2p_idx], epoch=epoch, tot_epochs=tot_epochs)
            true_token_count, all_token_count = [i + j for i, j in zip(p2s_true_token, s2p_true_token)], \
                [i + j for i, j in zip(p2s_all_token, s2p_all_token)]
            p2s_loss, s2p_loss = p2s_loss.sum(dim=-1), s2p_loss.sum(dim=-1)
            if sub_loss is not None:
                p2s_subloss, s2p_subloss = sub_loss[p2s_idx], sub_loss[s2p_idx]
                p2s_loss, s2p_loss = p2s_loss + p2s_subloss, s2p_loss + s2p_subloss
            p2s_loss, s2p_loss = p2s_loss.mean(), s2p_loss.mean()
            main_loss = p2s_loss + s2p_loss
        else:  # when model training on USPTO_MIT without dual-task, the loss and accuracy for reaction outcome prediction will also be shown in "p2s".
            p2s_loss, p2s_true_token, p2s_all_token = self.main_criterion(input=dec_out, target=data.tgt_seq,
                                                                          seq_len=data.tgt_seq_len, epoch=epoch, tot_epochs=tot_epochs)
            p2s_loss = p2s_loss.sum(dim=-1)
            true_token_count, all_token_count = p2s_true_token, p2s_all_token
            if sub_loss is not None:
                p2s_loss = p2s_loss + sub_loss
            p2s_loss = p2s_loss.mean()
            main_loss = p2s_loss

        # add kl_loss
        # main_loss += self.kl_weight * kld_loss

        # TODO: 之后可以返回一个 kl loss
        return main_loss, true_token_count, all_token_count, sub_loss_record

    def gen_acc(
            self,
            dec_out: torch.Tensor,
            data: G2SBatchData
    ):
        with torch.no_grad():
            dec_pred = torch.softmax(dec_out, dim=-1).argmax(dim=-1)
            pred_mask1 = data.tgt_seq.ne(self.token_idx['<PAD>']).to(torch.long)
            pred_mask2 = data.tgt_seq.eq(self.token_idx['<PAD>']).to(torch.long)
            token_acc = ((dec_pred == data.tgt_seq).to(torch.float) * pred_mask1).sum(dim=-1) \
                        / pred_mask1.sum(dim=-1)
            seq_acc = torch.logical_or((dec_pred == data.tgt_seq), pred_mask2) \
                .all(dim=-1).to(torch.float)
            if self.use_subs:
                p2s_idx, s2p_idx = data.bi_label.eq(0), data.bi_label.eq(1)
                token_acc = [token_acc[p2s_idx].mean().item(), token_acc[s2p_idx].mean().item()]
                seq_acc = [seq_acc[p2s_idx].mean().item(), seq_acc[s2p_idx].mean().item()]
            else:
                token_acc = [token_acc.mean().item(), 0]
                seq_acc = [seq_acc.mean().item(), 0]
        return token_acc, seq_acc


    def forward(
            self,
            data: G2SBatchData,
            epoch,
            tot_epochs
    ):  # a simple forward step, return loss and token accuracy.
        batch_size = len(data.src_graph_len)
        enc_mem = self.encode(data)
        # enc_ref_mem = self.encode_ref(data)     # TODO: 遇到 NoneType

        bos = torch.full((batch_size, 1), self.token_idx['<BOS>'],
                         dtype=torch.long, device=data.tgt_seq.device)
        tgt_seq = torch.cat([bos, data.tgt_seq], dim=1)
        tgt_seq = self.seq_embedding(tgt_seq)

        # TODO: 这里可以嵌入 CVAE
        Kld_loss = 0
        y_sample, Kld_loss = self.cvae(enc_mem, tgt_seq)
        # y_sample = y_sample.argmax(-1).long()

        # 直接融到 bos 里
        latent = y_sample @ self.cls_embedding.weight
        # lv_rxn_emb = latent * math.sqrt(self.d_model)  # .unsqueeze(1)
        tgt_seq_clone = tgt_seq.clone()
        tgt_seq_clone[:, 0, :] += latent
        tgt_seq = tgt_seq_clone

        cls_cache = None
        if self.cls_len > 0:
            if self.use_subs:
                cls_cache = self.bi_embedding(data.bi_label).unsqueeze(1)
            if self.use_reaction_type:
                cls_cache = self.cls_embedding(data.reaction_type - 1).unsqueeze(1) if \
                    cls_cache is None else torch.cat(
                    [self.cls_embedding(data.reaction_type - 1).unsqueeze(1), cls_cache], dim=1)
            # if self.model_lat_rxn:
            #     # cls_cache = latent.unsqueeze(1) if \
            #     #     cls_cache is None else torch.cat([latent.unsqueeze(1), cls_cache], dim=1)
            #
            #     cls_cache = self.cls_embedding(y_sample).unsqueeze(1) if \
            #         cls_cache is None else torch.cat([self.cls_embedding(y_sample).unsqueeze(1), cls_cache], dim=1)

        if cls_cache is not None:
            tgt_seq = torch.cat([cls_cache, tgt_seq], dim=1) * math.sqrt(self.d_model)
        else:
            # add latent rxn type as new token RSMNorm?
            tgt_seq = tgt_seq * math.sqrt(self.d_model)
        tgt_seq = self.dec_dropout(tgt_seq)

        self.transformer_decoder._init_cache(
            context_cache=enc_mem
        )
        dec_mem = self.transformer_decoder(
            tgt=tgt_seq,
            context_len=[data.src_graph_len],
            tgt_len=[data.tgt_seq_len + 1 + self.cls_len],      # +1: bos, cls_len: additional cls
            task=data.task,
            rel=None,
            deg=None,
            future=True
        )
        dec_out = dec_mem[:, self.cls_len:-1]
        if self.use_subs:
            if data.task == 'bidirection':
                dec_out = torch.chunk(dec_out, 2, dim=0)
                dec_out = torch.cat([self.out_project[0](dec_out[0]), self.out_project[1](dec_out[1])], dim=0)
            elif data.task == 'prod2subs':
                dec_out = self.out_project[0](dec_out)
            elif data.task == 'subs2prod':
                dec_out = self.out_project[1](dec_out)
        else:
            dec_out = self.out_project(dec_out)

        main_loss, true_token_count, all_token_count, sub_loss = self.gen_loss(
            dec_out=dec_out,
            data=data,
            kld_loss=Kld_loss,
            epoch=epoch,
            tot_epochs=tot_epochs
        )
        token_acc, seq_acc = self.gen_acc(
            dec_out=dec_out,
            data=data
        )
        return main_loss, token_acc, seq_acc, true_token_count, all_token_count, sub_loss, Kld_loss, 0  # kl_disc = 0


    def gen_predict(
            self,
            data: G2SBatchData,
            args,
            beam_size = 10
    ):
        batch_size = len(data.src_graph_len)
        beam_module = args.beam_module
        # beam_size = args.beam_size

        enc_mem = self.encode(data)
        cur_batch_size = enc_mem.shape[0]
        enc_mem_cp = enc_mem.clone()

        # use latent rxn type
        cls_cache = None
        if self.use_reaction_type and self.cls_len > 0:
            cls_idx = data.reaction_type - 1
            cls_cache = self.cls_embedding(cls_idx).repeat_interleave(beam_size, dim=0).unsqueeze(1)
        # if self.model_lat_rxn and self.cls_len > 0:
        #     cls_cache = y_sample @ self.cls_embedding.weight
        #     cls_cache = cls_cache.unsqueeze(1)
            # cls_cache = self.cls_embedding(y_sample).repeat_interleave(beam_size, dim=0).unsqueeze(1)

        enc_len = torch.repeat_interleave(data.src_graph_len, beam_size, dim=0)
        enc_mem = torch.repeat_interleave(enc_mem, beam_size, dim=0)

        # variational inference
        beam_y_sample = self.cvae.get_prior(enc_mem)
        beam_rxn_label = beam_y_sample.argmax(-1).to(self.device)

        # 转换为 [batch_size, beam_size] 形式
        # beam_rxn_label = beam_rxn_label.view(-1, beam_size)

        assert beam_module in ['huggingface', 'OpenNMT']
        if beam_module == 'huggingface':
            # beam_search = Lv_Huggingface_Beam(
            beam_search=Huggingface_Beam(
                beam_size=beam_size,
                batch_size=batch_size,
                bos_token_ids=self.token_idx['<BOS>'],
                pad_token_ids=self.token_idx['<PAD>'],
                eos_token_ids=self.token_idx['<EOS>'],
                length_penalty=0.,
                min_len=1,
                max_len=args.max_len,
                beam_group=1,
                temperature=args.T,
                top_k=args.k_sample,
                top_p=args.p_sample,
                return_num=args.return_num,
                remove_finish_batch=True,
                device=self.device
            )
        elif beam_module == 'OpenNMT':
            beam_search = Lv_OpenNMT_Beam(
                beam_size=beam_size,
                batch_size=batch_size,
                bos_token_ids=self.token_idx['<BOS>'],
                pad_token_ids=self.token_idx['<PAD>'],
                eos_token_ids=self.token_idx['<EOS>'],
                unk_token_ids=self.token_idx['<UNK>'],
                length_penalty=0.,
                min_len=1,
                max_len=args.max_len,
                temperature=args.T,
                top_k=args.k_sample,
                top_p=args.p_sample,
                return_num=args.return_num,
                device=self.device
            )
            beam_search._prepare(
                memory_bank=enc_mem_cp.transpose(0, 1),
                src_lengths=data.src_graph_len,
                src_map=None,
                target_prefix=None,
                initial_rxn_label=beam_rxn_label
            )


        # track beam during search
        # tracked_beam_rxn_label = beam_rxn_label.clone()

        lv_rxn_emb = beam_y_sample @ self.cls_embedding.weight

        if self.use_subs and self.cls_len > 0:
            bi_emb = self.bi_embedding(data.bi_label)
            bi_emb = torch.repeat_interleave(bi_emb, beam_size, dim=0).unsqueeze(1)
            if cls_cache is not None:
                cls_cache = torch.cat([cls_cache, bi_emb], dim=1)
            else:
                cls_cache = bi_emb

        self.transformer_decoder._init_cache(
            context_cache=enc_mem
        )

        for step in range(args.max_len):
            # tgt_seq: [batch_size, beam_size]
            tgt_seq = beam_search.current_token.reshape(-1, 1).to(self.device)
            tgt_seq = self.seq_embedding(tgt_seq)
            if step == 0:
                tgt_seq_clone = tgt_seq.clone()
                tgt_seq_clone[:, 0, :] += lv_rxn_emb
                tgt_seq = tgt_seq_clone

                # incorporate lv_rxn on bos
                tgt_seq = torch.cat([cls_cache, tgt_seq], dim=1) * math.sqrt(self.d_model) if cls_cache is not None \
                    else tgt_seq * math.sqrt(self.d_model)
            else:
                tgt_seq = tgt_seq * math.sqrt(self.d_model)
            dec_mem = self.transformer_decoder(
                tgt=tgt_seq,
                context_len=[enc_len],
                tgt_len=None,
                task=data.task,
                rel=None,
                deg=None,
                future=False,
                step=step
            )
            if step == 0 and self.cls_len > 0:
                dec_out = dec_mem[:, -1].unsqueeze(1)
            else:
                dec_out = dec_mem
            if self.use_subs:
                if data.task == 'bidirection':
                    dec_out = torch.chunk(dec_out, 2, dim=0)
                    dec_out = torch.cat([self.out_project[0](dec_out[0]), self.out_project[1](dec_out[1])], dim=0)
                elif data.task == 'prod2subs':
                    dec_out = self.out_project[0](dec_out)
                elif data.task == 'subs2prod':
                    dec_out = self.out_project[1](dec_out)
            else:
                dec_out = self.out_project(dec_out)

            # dec_out: probability of current token in the beam
            beam_search.generate(dec_out.to(self.device), 0)
            if beam_search.is_done: break
            # beam_search.mem_ids: [batch_size * beam_size]
            select_idx = beam_search.mem_ids.to(self.device)
            self.transformer_decoder._update_cache(select_idx)
            enc_len = torch.index_select(enc_len, dim=0, index=select_idx)

            # tracked_beam_rxn_label = torch.index_select(tracked_beam_rxn_label, dim=0, index=select_idx)

        predicted_rxn_labels_for_seq = torch.zeros((cur_batch_size, beam_size), dtype=torch.long)

        if beam_module == 'OpenNMT':
            predict_seq, predict_score, prob_per_token, predicted_rxn_labels_for_seq = beam_search.finish_generate()
        else:
            predict_seq, predict_score = beam_search.finish_generate()
            # predict_seq, predict_score = beam_search.finish_generate(self.batch_size, self.n_best)
            # redict_seq, predict_score, predicted_rxn_labels_for_seq = beam_search.finish_generate(
            #     beam_rxn_label_for_tracking=tracked_beam_rxn_label.reshape(cur_batch_size, beam_size)  # 确保维度匹配
            # )

        # select_idx = beam_search.mem_ids.to(self.device)
        # selected_rxn_type = torch.index_select(beam_rxn_label, dim=0, index=select_idx)

        return predict_seq, predict_score, predicted_rxn_labels_for_seq   # , y_sample, prob_per_token


    def update_kl_weight(self, args, anneal_rate):
        if args.vae_warmup > 0:
            self.kl_weight = min(1.0, self.kl_weight + anneal_rate)


    def encode_pairs(self, data: G2SBatchData):
        """
        input: product graphs and reactant SMILES strings
        return: embeddings of products and reactants, as well as inferred reaction types
        """

        batch_size = len(data.src_graph_len)
        enc_mem = self.encode(data)
        # enc_ref_mem = self.encode_ref(data)     # TODO: 遇到 NoneType

        bos = torch.full((batch_size, 1), self.token_idx['<BOS>'],
                         dtype=torch.long, device=data.tgt_seq.device)
        tgt_seq = torch.cat([bos, data.tgt_seq], dim=1)
        tgt_seq = self.seq_embedding(tgt_seq)

        y_sample, _ = self.cvae(enc_mem, tgt_seq)
        rxn_labels = y_sample.argmax(-1)

        # 直接融到 bos 里
        latent = y_sample @ self.cls_embedding.weight
        # lv_rxn_emb = latent * math.sqrt(self.d_model)  # .unsqueeze(1)
        tgt_seq_clone = tgt_seq.clone()
        tgt_seq_clone[:, 0, :] += latent
        tgt_seq = tgt_seq_clone

        cls_cache = None
        if self.cls_len > 0:
            if self.use_subs:
                cls_cache = self.bi_embedding(data.bi_label).unsqueeze(1)
            if self.use_reaction_type:
                cls_cache = self.cls_embedding(data.reaction_type - 1).unsqueeze(1) if \
                    cls_cache is None else torch.cat(
                    [self.cls_embedding(data.reaction_type - 1).unsqueeze(1), cls_cache], dim=1)
            # if self.model_lat_rxn:
            #     # cls_cache = latent.unsqueeze(1) if \
            #     #     cls_cache is None else torch.cat([latent.unsqueeze(1), cls_cache], dim=1)
            #
            #     cls_cache = self.cls_embedding(y_sample).unsqueeze(1) if \
            #         cls_cache is None else torch.cat([self.cls_embedding(y_sample).unsqueeze(1), cls_cache], dim=1)

        if cls_cache is not None:
            tgt_seq = torch.cat([cls_cache, tgt_seq], dim=1) * math.sqrt(self.d_model)
        else:
            # add latent rxn type as new token RSMNorm?
            tgt_seq = tgt_seq * math.sqrt(self.d_model)
        tgt_seq = self.dec_dropout(tgt_seq)

        self.transformer_decoder._init_cache(
            context_cache=enc_mem
        )
        dec_mem = self.transformer_decoder(
            tgt=tgt_seq,
            context_len=[data.src_graph_len],
            tgt_len=[data.tgt_seq_len + 1 + self.cls_len],  # +1: bos, cls_len: additional cls
            task=data.task,
            rel=None,
            deg=None,
            future=True
        )
        dec_out = dec_mem[:, self.cls_len:-1]
        if self.use_subs:
            if data.task == 'bidirection':
                dec_out = torch.chunk(dec_out, 2, dim=0)
                dec_out = torch.cat([self.out_project[0](dec_out[0]), self.out_project[1](dec_out[1])], dim=0)
            elif data.task == 'prod2subs':
                dec_out = self.out_project[0](dec_out)
            elif data.task == 'subs2prod':
                dec_out = self.out_project[1](dec_out)
        else:
            dec_out = self.out_project(dec_out)
        ###########

        # mean_src = torch.mean(latent, dim=1)
        # mean_tgt = torch.mean(dec_out, dim=1)
        # mean_pair = torch.cat([mean_src, mean_tgt], dim=-1)
        # mean_pair = torch.cat([latent, mean_tgt], dim=-1)

        print(f'rxn_labels.shape: {rxn_labels.shape}, y_sample.shape = {y_sample.shape}')
        assert rxn_labels.dim() == 1, "rxn should be 1D tensors"

        # return mean_pair, rxn_labels

        # print(f"latent.shape = {latent.shape}")
        # assert 1 ==2
        return latent, rxn_labels