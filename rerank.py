import argparse
import os
import random
import numpy as np
import pandas as pd
import json
import torch
from tqdm import trange
from copy import deepcopy
from preprocess_utils import get_vocab_size, get_char_to_ix, get_ix_to_char
# from modeling import TransformerConfig, Transformer, get_products_mask, get_reactants_mask, get_mutual_mask
from rdkit import Chem
from rdkit.rdBase import DisableLog
from reward_model import RewardTransformerConfig, RewardTransformer, get_input_mask_reward, get_output_mask_reward, \
    get_mutual_mask_reward
from typing import Dict, List
import math


# for Retrosynthesis model
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.parsing import get_parser, post_setting_args
from utils.chem_tools import NODE_FDIM, BOND_FDIM
from utils.data_loader import G2SFullDataset, S2SFullDataset
from utils.wrap_single_smi import InferBatch, InferBatch_wo_rxns

from models.graph_rel_transformers import Graph_Transformer_Base
from models.graph_vae_transformers import Graph_Transformer_VAE
from models.seq_rel_transformers import Seq_Transformer_Base
from models.seq_vae_transformers import Seq_Transformer_VAE
from models.Graph_LWLV_SMILES import Graph_lv_Transformer_Base
from models.SMILES_LWLV_SMILES import Seq_lv_Transformer_Base
from models.Graph_Lv_Rxn_SMILES import Graph_lv_rxn_Transformer_Base

from models.module_utils import Model_Save, eval_plot, coverage_plot, beam_result_process, process_multi_rxn_coverage, beam_result_process_1_to_N

proj_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_dir, 'preprocessed')
origin_dir = os.path.join(proj_dir, 'data')

DisableLog('rdApp.warning')


list_smi_uncano = []
list_smi_cano = []

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles


# modified by Neuralsym
class Proposer:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Infer_wrapper = InferBatch_wo_rxns('./preprocessed', args)
        self.model = self.build_model(args)
        self.model.model.eval()
        self.return_size = args.beam_size

    def build_model(self, args):

        if args.use_subs and args.use_reaction_type:
            dec_cls = 2
        elif args.use_subs or args.use_reaction_type:
            dec_cls = 1
        else:
            dec_cls = 0

        ckpt_dir = os.path.join('checkpoints', args.save_name)
        token_idx = self.Infer_wrapper.token_index
        module_saver = Model_Save(
            ckpt_dir=ckpt_dir,
            device=args.device,
            save_strategy=args.save_strategy,
            save_num=args.save_num,
            swa_count=args.swa_count,
            swa_tgt=args.swa_tgt,
            const_save_epoch=args.const_save_epoch,
            top1_weight=args.top1_weight
        )
        # newly setted model
        if args.model_type == 'G2_LW_RXN_CVAT':
            module = Graph_lv_rxn_Transformer_Base(
                f_vocab=len(token_idx),
                f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
                f_bond=BOND_FDIM,
                token_idx=token_idx,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        elif args.model_type == 'G2_LW_CVAT':
            predict_module = Graph_lv_Transformer_Base(
                f_vocab=len(token_idx),
                f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
                f_bond=BOND_FDIM,
                token_idx=token_idx,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        elif args.model_type == 'S2S_LW_CVAT':
            predict_module = Seq_lv_Transformer_Base(
                f_vocab=len(token_idx),
                token_idx=token_idx,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        elif args.model_type == 'BiG2S':
            predict_module = Graph_Transformer_Base(
                f_vocab=len(token_idx),
                f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
                f_bond=BOND_FDIM,
                token_idx=self.Infer_wrapper.token_index,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        elif args.model_type == 'BiG2S_HCVAE_RXN':
            predict_module = Graph_Transformer_VAE(
                f_vocab=len(token_idx),
                f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
                f_bond=BOND_FDIM,
                token_idx=self.Infer_wrapper.token_index,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        elif args.model_type == 'S2S_HCVAE_RXN':
            predict_module = Seq_Transformer_VAE(
                f_vocab=len(token_idx),
                token_idx=self.Infer_wrapper.token_index,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )
        else:  # pure transformer : S2S
            predict_module = Seq_Transformer_Base(
                f_vocab=len(token_idx),
                token_idx=self.Infer_wrapper.token_index,
                token_freq=None,
                token_count=None,
                cls_len=dec_cls,
                args=args
            )

        _, predict_module.model, _, _ = module_saver.load(args.ckpt_list[0], predict_module.model)
        return predict_module

    def propose(self,
                smi:str,
                topk: int = 10,
                **kwargs) -> List[Dict[str, List]]:
        answer = []
        aim_size = topk
        with torch.no_grad():
            batch_graph_input = self.Infer_wrapper.preprocess(smi, rxn_type=1)      # ERROR: 遇到了词表里没有的token
            batch_data = batch_graph_input.to(self.args.device)
            # predict_scores = logp(React|Prod)
            predict_result, predict_scores, _ = self.model.model_predict(
                data=batch_data,
                args=self.args,
                beam_size=10        # 后面要筛出有效分子，如果和 top-k 一样怕生成出的有无效分子
            )
            # output content: f"RX{self.rxn_type}_TOP{beam_id + 1},{res_smi}"
            # TODO: predict score 归一化，变正数
            smi_nodes_sorted, _ = self.Infer_wrapper.post_process(predict_result, predict_scores)

            # exp_pred_scores = torch.exp(predict_scores)
            # normed_pred_scores = torch.softmax(exp_pred_scores, dim=1)

            # smi_nodes_sorted & predict_scores are all in [1, beam_size], should be transformed to [beam_size]
            # for i, (pred_react, score) in enumerate(zip(smi_nodes_sorted[0], normed_pred_scores[0])):
            for i, (pred_react, score) in enumerate(zip(smi_nodes_sorted[0], predict_scores[0])):
                # delete 'RX{self.rxn_type}_TOP{beam_id + 1}'
                pred_react = pred_react.split(',')[-1]
                reactants = set(pred_react.split("."))
                num_valid_reactant = 0
                sms = set()
                for r in reactants:
                    m, cano_r = cano_smiles(r)
                    if m is not None:
                        num_valid_reactant += 1
                        # m = Chem.RemoveHs(m)
                        if cano_r != r:        # TODO: 只要结果变了，就用原来的，避免换成某些oov的token
                            # print(f'react before canolize: {r}')
                            # print(f'react after canolize: {cano_r}')
                            # sms.add(r)  # canonicalize ？
                            list_smi_cano.append(cano_r)
                            list_smi_uncano.append(r)
                        # assert 1 == 2
                        sms.add(cano_r)
                if num_valid_reactant != len(reactants):
                    continue
                if len(sms):
                    # error: score < 0 非法，直接拿来当结果
                    # answer.append([sorted(list(sms)), -math.log10(score + 1e-10)])  # Tuple[precs, score] where precs is a List[str]
                    # score = -log p, p 越大 score 越小
                    len_smis = len(pred_react)
                    if self.args.norm_score:
                        answer.append([sorted(list(sms)), -1.0 * score.item() / len_smis])       # TODO: 要不要把 value 从 tensor 里拿出来？
                    else:
                        answer.append([sorted(list(sms)), -1.0 * score.item()])
                    aim_size -= 1
                if aim_size == 0:
                    break

        # print(answer)

        return answer[:topk]


def convert_symbols_to_inputs(products_list, reactants_list, max_length):
    num_samples = len(products_list)
    # products
    products_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    products_input_mask = torch.zeros((num_samples, max_length), device=device)

    # reactants
    reactants_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    reactants_input_mask = torch.zeros((num_samples, max_length), device=device)

    for cnt in range(num_samples):
        products = '^' + products_list[cnt] + '$'
        reactants = '^' + reactants_list[cnt] + '$'

        for i, symbol in enumerate(products):
            products_input_ids[cnt, i] = char_to_ix[symbol]
        products_input_mask[cnt, :len(products)] = 1

        for i in range(len(reactants) - 1):
            reactants_input_ids[cnt, i] = char_to_ix[reactants[i]]
        reactants_input_mask[cnt, :len(reactants) - 1] = 1
    return (products_input_ids, products_input_mask, reactants_input_ids, reactants_input_mask)


def load_dataset(split):
    file_name = "../%s_dataset.json" % split       # Note! canonicalize 后部分原子的属性会变化，导致tokenize后的token变化
    file_name = os.path.expanduser(file_name)
    dataset = []  # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
            _, product = cano_smiles(product)
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees']) + 1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list,
                "depth": reaction_trees['depth']
            })

    return dataset


def convert_symbols_to_inputs_reward(input_list, output_list, max_length):
    num_samples = len(input_list)       # number of products
    # input
    input_ids = np.zeros((num_samples, max_length))
    input_mask = np.zeros((num_samples, max_length))

    # output
    output_ids = np.zeros((num_samples, max_length))
    output_mask = np.zeros((num_samples, max_length))

    # for output
    token_ids = np.zeros((num_samples, max_length))
    token_mask = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        input_ = '^' + input_list[cnt] + '$'
        output_ = '^' + output_list[cnt] + '$'

        for i, symbol in enumerate(input_):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input_)] = 1

        for i in range(len(output_) - 1):
            output_ids[cnt, i] = char_to_ix[output_[i]]
            token_ids[cnt, i] = char_to_ix[output_[i + 1]]
            if i != len(output_) - 2:       # 当 i == len(output_) - 2 时，token_ids[cnt, i] = $
                token_mask[cnt, i] = 1
        output_mask[cnt, :len(output_) - 1] = 1

    return (input_ids, input_mask, output_ids, output_mask, token_ids, token_mask)


def get_rerank_scores(input_list, output_list):
    """
    input_list: products
    output_list: material reactants
    """
    if input_list:
        longest_input = max(input_list, key=len)
        max_length_input = len(longest_input)           # max length of SMILES strings of products
    else:
        max_length_input = 0
    if output_list:
        longest_output = max(output_list, key=len)
        max_length_output = len(longest_output)         # max length of SMILES strings of material reactants
    else:
        max_length_output = 0
    max_length_reward = max(max_length_input, max_length_output) + 2        # max length of above molecules
    (input_ids,
     input_mask,
     output_ids,
     output_mask,
     token_ids,
     token_mask) = convert_symbols_to_inputs_reward(input_list, output_list, max_length_reward)

    # torch.tensor(input_ids, dtype=torch.long, device=device)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    input_mask = torch.tensor(input_mask, dtype=torch.float32, device=device)
    output_ids = torch.tensor(output_ids, dtype=torch.long, device=device)
    output_mask = torch.tensor(output_mask, dtype=torch.float32, device=device)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_mask = torch.tensor(token_mask, dtype=torch.float32, device=device)

    # input_ids = torch.LongTensor(input_ids).to(device)
    # input_mask = torch.FloatTensor(input_mask).to(device)
    # output_ids = torch.LongTensor(output_ids).to(device)
    # output_mask = torch.FloatTensor(output_mask).to(device)
    # token_ids = torch.LongTensor(token_ids).to(device)
    # token_mask = torch.FloatTensor(token_mask).to(device)

    mutual_mask = get_mutual_mask_reward([output_mask, input_mask])
    input_mask = get_input_mask_reward(input_mask)
    output_mask = get_output_mask_reward(output_mask)

    # 分batch
    with torch.no_grad():
        logits = reward_model(input_ids, output_ids, input_mask, output_mask, mutual_mask)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=token_ids.unsqueeze(2)).squeeze(2)      # [n_sample, seq_len_reactants]

        all_logps = (per_token_logps * token_mask).sum(-1) / token_mask.sum(-1)     # 以每个token的logp加和作为对整个SMILES串的打分

    torch.cuda.empty_cache()
    return all_logps


def get_rerank_scores_batch(input_list, output_list, batch_size=128):
    """
    input_list: list of product SMILES
    output_list: list of reactant SMILES
    batch_size: how many samples to process in one forward pass
    returns: tensor of scores, shape [n_samples]
    """
    if input_list:
        longest_input = max(input_list, key=len)
        max_length_input = len(longest_input)
    else:
        max_length_input = 0
    if output_list:
        longest_output = max(output_list, key=len)
        max_length_output = len(longest_output)
    else:
        max_length_output = 0
    max_length_reward = max(max_length_input, max_length_output) + 2

    n_samples = len(input_list)
    all_logps_list = []

    for start_idx in trange(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_inputs = input_list[start_idx:end_idx]
        batch_outputs = output_list[start_idx:end_idx]

        (input_ids,
         input_mask,
         output_ids,
         output_mask,
         token_ids,
         token_mask) = convert_symbols_to_inputs_reward(batch_inputs, batch_outputs, max_length_reward)

        input_ids = torch.LongTensor(input_ids).to(device)
        input_mask = torch.FloatTensor(input_mask).to(device)
        output_ids = torch.LongTensor(output_ids).to(device)
        output_mask = torch.FloatTensor(output_mask).to(device)
        token_ids = torch.LongTensor(token_ids).to(device)
        token_mask = torch.FloatTensor(token_mask).to(device)
        mutual_mask = get_mutual_mask_reward([output_mask, input_mask])
        input_mask = get_input_mask_reward(input_mask)
        output_mask = get_output_mask_reward(output_mask)
        with torch.no_grad():
            logits = reward_model(input_ids, output_ids, input_mask, output_mask, mutual_mask)
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=token_ids.unsqueeze(2)).squeeze(2)

            all_logps = (per_token_logps * token_mask).sum(-1) / token_mask.sum(-1)
            all_logps_list.append(all_logps)
        torch.cuda.empty_cache()

    res_all_logps = torch.cat(all_logps_list, dim=0)
    return res_all_logps


def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True


def get_route_result(task, proposer):
    max_depth = task["depth"]
    # Initialization
    answer_set = []
    queue = []
    queue.append({
        "score": 0.0,
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })

    # record SMILES that changes tokens during canonicalization

    while True:
        if len(queue) == 0:
            break
        nxt_queue = []
        for item in queue:
            score = item["score"]
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            if depth > max_depth:
                continue
            # print(f"first_route = {first_route}")

            # cur_solutions = proposer.propose(smi=first_route[-1])
            # print(f"cur_solutions = {cur_solutions}")       # error: None
            #
            # assert 1 == 2

            # first_route[-1] = 'Cc1ccccc1[SH]1(=O)CCN(C(=O)OC(C)(C)C)CC1'
            cano_prod = first_route[-1]
            # TODO: only used in debugging
            # cano_prod = 'CSC(SC)=C(Cl)c1ccc(CC(C)C)cc1'
            # cano_prod = 'CSC(=Cc1ccc(CC(C)C)cc1)S(C)C'

            # print(f"current input prod: {cano_prod}")
            for expansion_solution in proposer.propose(smi=cano_prod):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, expansion_score = expansion_solution[0], expansion_solution[1]
                # reactants 被拆开了
                expansion_reactants = sorted(expansion_reactants)
                # 找到了原料，放到最终答案里，且不存到nxt_queue中
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score + expansion_score,
                        "starting_materials": iter_starting_materials + expansion_reactants,
                    })
                else:
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)            # end_point
                        else:
                            # intermediate point
                            iter_routes = [{"route": first_route + [reactant], "depth": depth + 1}] + iter_routes

                    nxt_queue.append({
                        "score": score + expansion_score,               # int
                        "routes_info": iter_routes,                     # list
                        "starting_materials": iter_starting_materials   # list
                    })
        # 根据score升序排序，且仅维护前5个
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]

    # 根据score升序排序

    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()      # inchi-keys
    final_answer_set = []       # score, inchi-keys
    rerank_input_list = []      # product
    rerank_output_list = []     # material reactants
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]     # list of material reactants

        cano_starting_materials = []
        for material_ in starting_materials:
            _, cano_material_ = cano_smiles(material_)
            cano_starting_materials.append(cano_material_)

        answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
            final_answer_set.append({
                "score": score,
                "answer_keys": answer_keys
            })
            rerank_input_list.append(task['product'])
            rerank_output_list.append('.'.join(sorted(cano_starting_materials)))

    # 重新评分 -- 基于CREBM的重排？
    # 全部拿去预测会爆显存，拆开算
    rerank_scores = get_rerank_scores(rerank_input_list, rerank_output_list)        # [n_samples, 1]
    # rerank_scores = get_rerank_scores_batch(rerank_input_list, rerank_output_list)

    norm_coeff = rerank_scores.sum() * -1

    # rerank_scores = torch.cat([rerank_tmp_1_scores, rerank_tmp_2_scores], dim=0)
    for i, score_ in enumerate(rerank_scores):
        final_answer_set[i]["rerank_score"] = -score_.item()        # 预测结果为 log < 0, 意义是啥呢
        # 重排后的结果 ~ -(logP - E_\theta(X, Y))
        final_answer_set[i]["total_score"] = -args.alpha * score_.item() + final_answer_set[i]["score"]      # args.alpha
    final_answer_set = sorted(final_answer_set, key=lambda x: x["total_score"])[:args.beam_size]

    # ablation results
    original_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:args.beam_size]

    return final_answer_set, original_answer_set


def check_hit_ground_truths(task, final_answer_set, original_answer_set):
    # Calculate answers for both reranked routes and original routes
    max_depth = task["depth"]
    ground_truth_keys_list = [
        set([
            Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
        ]) for targets in task["targets"]
    ]

    res_rerank_depth = task["depth"]
    res_rerank_topk = None
    res_ori_depth = task["depth"]
    res_ori_topk = None

    found = False
    rerank_result = None

    # for rerank routes
    for rank, answer in enumerate(final_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                # return max_depth, rank      # rank 就是 top-k
                rerank_result = (max_depth, rank)  # 暂存 max_depth 和 rank
                found = True  # 设置标志变量为 True
                break  # 退出内层循环
            if found:
                break  # 退出外层循环
    if rerank_result:
        res_rerank_depth, res_rerank_topk = rerank_result[0], rerank_result[1]

    found = False
    ori_result = None
    # for original routes
    for rank, answer in enumerate(original_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                # return max_depth, rank      # rank 就是 top-k
                ori_result = (max_depth, rank)  # 暂存 max_depth 和 rank
                found = True  # 设置标志变量为 True
                break  # 退出内层循环
            if found:
                break  # 退出外层循环
    if ori_result:
        res_ori_depth, res_ori_topk = ori_result[0], ori_result[1]

    return res_rerank_depth, res_rerank_topk, res_ori_depth, res_ori_topk


if __name__ == "__main__":
    parser = get_parser(mode = 'test')


    # arguments for reward model
    # parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
    parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
    parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Number of layers in encoder\'s module. Default 3.')
    parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
    parser.add_argument('--intermediate_size', type=int, default=512,
                        help='The size of hidden units of position-wise layer.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for decoding. Default 1.2')
    # parser.add_argument('--beam_size', type=int, default=5,
    #                     help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
    parser.add_argument("--alpha", type=float, default=0.01)

    # adjust score norm
    parser.add_argument("--norm_score", action="store_true")
    # load result file?
    parser.add_argument("--load_res", action="store_true")

    args = post_setting_args(parser)

    args.beam_size = 5      # Default 5. Must be 1 meaning greedy search or greater or equal 5.
    print(f"beam_size = {args.beam_size}")

    # init retrosynthesis model
    RetroProposer = Proposer(args)

    # init reward model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()

    stock = pd.read_hdf('../zinc_stock_17_04_20.hdf5', key="table")
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    reward_config = RewardTransformerConfig(vocab_size=vocab_size,
                                            embedding_size=64,
                                            hidden_size=512,
                                            num_hidden_layers=6,
                                            num_attention_heads=8,
                                            intermediate_size=1024,
                                            hidden_dropout_prob=0.1)
    reward_model = RewardTransformer(reward_config)
    checkpoint = torch.load("pretrain_reward_model.pkl")
    reward_model.load_state_dict(checkpoint.state_dict())
    reward_model.to(device)
    reward_model.eval()

    tasks = load_dataset('test')
    # for original routes
    overall_result = np.zeros((args.beam_size, 2))              # 0: results that hit the ground truths
    depth_hit = np.zeros((2, 15, args.beam_size))               # 1: total results
    # for rerank routes
    rerank_overall_result = np.zeros((args.beam_size, 2))       # 0: results that hit the ground truths
    rerank_depth_hit = np.zeros((2, 15, args.beam_size))        # 1: total results

    # 存成 json文件，避免重复推理
    ori_route_file = f'checkpoints/{args.save_name}/beam_search_ori_routes.json'
    rerank_route_file = f'checkpoints/{args.save_name}/beam_search_reranked_routes.json'

    if not args.load_res:
        ori_route_dicts = defaultdict(list)
        rerank_route_dicts = defaultdict(list)
        # 一个task对应一个产物
        for epoch in trange(0, len(tasks)):
            final_answer_set, original_answer_set = get_route_result(tasks[epoch], proposer=RetroProposer)
            ori_route_dicts[epoch] = original_answer_set
            rerank_route_dicts[epoch] = final_answer_set

        with open(ori_route_file, 'w') as ori_f:
            json.dump(ori_route_dicts, ori_f, ensure_ascii=False, indent=4)
        with open(rerank_route_file, 'w') as rerank_f:
            json.dump(rerank_route_dicts, rerank_f, ensure_ascii=False, indent=4)  # 缩进长度
        print("Routes are saved!")

        compare_smis_file = 'compare_canolize_smis.txt'
        with open(compare_smis_file, 'w') as comp_f:
            for uncano_s, cano_s in zip(list_smi_uncano, list_smi_cano):
                smi_pair = uncano_s + ', ' + cano_s + '\n'
                comp_f.write(smi_pair)
    else:
        # load json
        with open(ori_route_file, 'r') as ori_f:
            ori_route_dicts = json.load(ori_f)
        with open(rerank_route_file, 'r') as rerank_f:
            rerank_route_dicts = json.load(rerank_f)
        for epoch in trange(0, len(tasks)):
            final_answer_set = rerank_route_dicts[str(epoch)]
            original_answer_set = ori_route_dicts[str(epoch)]
            res_rerank_depth, res_rerank_topk, res_ori_depth, res_ori_topk = check_hit_ground_truths(tasks[epoch],
                                                                                                     final_answer_set,
                                                                                                     original_answer_set)

            # for original routes
            overall_result[:, 1] += 1
            depth_hit[1, res_ori_depth, :] += 1
            if res_ori_topk is not None:
                overall_result[res_ori_topk:, 0] += 1
                depth_hit[0, res_ori_depth, res_ori_topk:] += 1
            # for reranked routes
            rerank_overall_result[:, 1] += 1
            rerank_depth_hit[1, res_rerank_depth, :] += 1
            if res_rerank_topk is not None:
                rerank_overall_result[res_rerank_topk:, 0] += 1
                rerank_depth_hit[0, res_rerank_depth, res_rerank_topk:] += 1

        print("original overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
        print("original depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])

        print("original overall_result: ", rerank_overall_result,
              100 * rerank_overall_result[:, 0] / rerank_overall_result[:, 1])
        print("original depth_hit: ", rerank_depth_hit, 100 * rerank_depth_hit[0, :, :] / rerank_depth_hit[1, :, :])
