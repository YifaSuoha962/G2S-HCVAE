import torch
import torch.nn as nn
import logging
import argparse
import random
import numpy as np
import os
import json
import pandas as pd
import math

from tqdm import trange
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from preprocess_utils import get_vocab_size, get_char_to_ix, get_ix_to_char
# from modeling import TransformerConfig, Transformer, get_products_mask, get_reactants_mask, get_mutual_mask
from rdkit.rdBase import DisableLog
from reward_model import (RewardTransformerConfig, RewardTransformer, get_input_mask_reward, get_output_mask_reward, get_mutual_mask_reward,
                          get_mutual_mask, get_products_mask, get_reactants_mask)

# retrog2s utils
from utils.parsing import get_parser, post_setting_args
from rerank import Proposer     # , get_rerank_scores

from collections import defaultdict

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


class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)     # np.bool 在 numpy.1.20.0之后被弃用了
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)

    return fp


def value_fn(smi):
    fp = smiles_to_fp(smi, fp_dim=args.fp_dim).reshape(1,-1)
    fp = torch.FloatTensor(fp).to(device)
    with torch.no_grad():
        v = value_model(fp).item()
    torch.cuda.empty_cache()
    return v


def convert_symbols_to_inputs(products_list, reactants_list, max_length):
    num_samples = len(products_list)
    #products
    products_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    products_input_mask = torch.zeros((num_samples, max_length), device=device)

    #reactants
    reactants_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    reactants_input_mask = torch.zeros((num_samples, max_length), device=device)

    for cnt in range(num_samples):
        products = '^' + products_list[cnt] + '$'
        reactants = '^' + reactants_list[cnt] + '$'
        
        for i, symbol in enumerate(products):
            products_input_ids[cnt, i] = char_to_ix[symbol]
        products_input_mask[cnt, :len(products)] = 1

        for i in range(len(reactants)-1):
            reactants_input_ids[cnt, i] = char_to_ix[reactants[i]]
        reactants_input_mask[cnt, :len(reactants)-1] = 1
    return (products_input_ids, products_input_mask, reactants_input_ids, reactants_input_mask)


def load_dataset(split):
    file_name = "../%s_dataset.json" % split
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
            _, product = cano_smiles(product)
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset

def convert_symbols_to_inputs_reward(input_list, output_list, max_length):
    num_samples = len(input_list)
    #input
    input_ids = np.zeros((num_samples, max_length))
    input_mask = np.zeros((num_samples, max_length))

    #output
    output_ids = np.zeros((num_samples, max_length))
    output_mask = np.zeros((num_samples, max_length))

    #for output
    token_ids = np.zeros((num_samples, max_length))
    token_mask = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        input_ = '^' + input_list[cnt] + '$'
        output_ = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input_):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input_)] = 1

        for i in range(len(output_)-1):
            output_ids[cnt, i] = char_to_ix[output_[i]]
            token_ids[cnt, i] = char_to_ix[output_[i+1]]
            if i != len(output_)-2:
                token_mask[cnt, i] = 1
        output_mask[cnt, :len(output_)-1] = 1
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


def get_rerank_scores_batch(input_list, output_list, batch_size=32):
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

    all_logps_accumulated = []  # 用于累积所有批次的结果

    # 计算总批次数
    num_samples = len(input_list)
    if num_samples == 0:
        return torch.tensor([], device=device)  # 如果没有样本，返回空张量

    num_batches = math.ceil(num_samples / batch_size)  # 向上取整以确保所有样本都被处理

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_input_list = input_list[start_idx:end_idx]
        batch_output_list = output_list[start_idx:end_idx]

        # 转换为模型输入格式
        (batch_input_ids,
         batch_input_mask,
         batch_output_ids,
         batch_output_mask,
         batch_token_ids,
         batch_token_mask) = convert_symbols_to_inputs_reward(batch_input_list, batch_output_list, max_length_reward)

        # 将列表转换为张量并移动到 GPU
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        input_mask_tensor = torch.tensor(batch_input_mask, dtype=torch.float32, device=device)
        output_ids_tensor = torch.tensor(batch_output_ids, dtype=torch.long, device=device)
        output_mask_tensor = torch.tensor(batch_output_mask, dtype=torch.float32, device=device)
        token_ids_tensor = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
        token_mask_tensor = torch.tensor(batch_token_mask, dtype=torch.float32, device=device)

        # 在每个批次处理前尝试清空缓存
        torch.cuda.empty_cache()  # 提前清空缓存

        mutual_mask_tensor = get_mutual_mask_reward([output_mask_tensor, input_mask_tensor])
        input_mask_tensor = get_input_mask_reward(input_mask_tensor)
        output_mask_tensor = get_output_mask_reward(output_mask_tensor)

        with torch.no_grad():
            # 将当前批次的数据送入模型
            logits = reward_model(input_ids_tensor, output_ids_tensor, input_mask_tensor, output_mask_tensor,
                                  mutual_mask_tensor)

            # 确保 token_ids_tensor 的维度与 logits.log_softmax(-1) 匹配
            # 可能是因为 index 需要匹配 [batch_size, seq_len] 的形状，unsqueeze(2) 是正确的
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=token_ids_tensor.unsqueeze(2)).squeeze(
                2)

            all_logps_batch = (per_token_logps * token_mask_tensor).sum(-1) / token_mask_tensor.sum(-1)
            all_logps_accumulated.append(all_logps_batch)

        # 每个批次处理完后再次清空缓存（如果需要在循环内部频繁释放）
        # 但通常一次循环结束后清空一次就够了
        # torch.cuda.empty_cache()

    # 将所有批次的结果拼接起来
    final_all_logps = torch.cat(all_logps_accumulated, dim=0)
    # print(f"final_all_logps.shape = {final_all_logps.shape}")
    # assert 1 == 2
    # 最终的缓存清理
    torch.cuda.empty_cache()

    return final_all_logps


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
        "score": value_fn(task["product"]),             # evaluate cost
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })
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
            expansion_mol = first_route[-1]
            cost_expansion_product = value_fn(expansion_mol)
            for expansion_solution in proposer.propose(expansion_mol, topk=proposer.return_size):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                # treat logP of predictions as cost
                expansion_reactants, reaction_cost = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score + reaction_cost - cost_expansion_product, # value_fn(expansion_mol),
                        "starting_materials": iter_starting_materials + expansion_reactants,
                    })
                else:
                    estimation_cost = 0
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            estimation_cost += value_fn(reactant)
                            iter_routes = [{"route": first_route + [reactant], "depth": depth + 1}] + iter_routes
                    nxt_queue.append({
                        "score": score + reaction_cost + estimation_cost - cost_expansion_product, # value_fn(expansion_mol),
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]

    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []
    rerank_input_list = []
    rerank_output_list = []
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]

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

    rerank_scores = get_rerank_scores_batch(rerank_input_list, rerank_output_list)
    # rerank_scores = get_rerank_scores(rerank_input_list, rerank_output_list)

    for i, score_ in enumerate(rerank_scores):
        final_answer_set[i]["rerank_score"] = -score_.item()
        final_answer_set[i]["total_score"] = -args.alpha * score_.item() + final_answer_set[i]["score"]
    final_answer_set = sorted(final_answer_set, key=lambda x: x["total_score"])[:args.beam_size]

    # ablation results
    original_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:args.beam_size]

    return final_answer_set, original_answer_set

    # Calculate answers
    # ground_truth_keys_list = [
    #     set([
    #         Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
    #     ]) for targets in task["targets"]
    # ]
    # for rank, answer in enumerate(final_answer_set):
    #     answer_keys = set(answer["answer_keys"])
    #     for ground_truth_keys in ground_truth_keys_list:
    #         if ground_truth_keys == answer_keys:
    #             return max_depth, rank
    #
    # return max_depth, None

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

    # args of crebm
    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_mlp_layers', type=int, default=1)

    # adjust score norm
    parser.add_argument("--norm_score", action="store_true")
    # load result file?
    parser.add_argument("--load_res", action="store_true")

    args = post_setting_args(parser)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Prepare modules for ranking
    # 1.1 init value model
    value_model = ValueMLP(
        n_layers=args.n_mlp_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1
    )
    value_model.load_state_dict(torch.load('value_mlp.pkl'))
    value_model.to(device)
    value_model.eval()

    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()

    stock = pd.read_hdf('../zinc_stock_17_04_20.hdf5', key="table")
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    # 1.2 rerank model -- CREBM
    reward_config = RewardTransformerConfig(vocab_size=vocab_size,
                                            embedding_size=64,
                                            hidden_size=512,
                                            num_hidden_layers=6,
                                            num_attention_heads=8,
                                            intermediate_size=1024,
                                            hidden_dropout_prob=0.1)
    reward_model = RewardTransformer(reward_config)
    checkpoint = torch.load("reward_model.pkl")
    reward_model.load_state_dict(checkpoint.state_dict())
    reward_model.to(device)
    reward_model.eval()

    # if args.model_type == 'BiG2S' :
    #     args.ckpt_list = ['swa4']
    #     args.T = 1.2
    # if args.model_type == 'BiG2S_HCVAE_RXN':
    #     args.ckpt_list = ['swa4']


    print(f"args.ckpt_list = {args.ckpt_list}")
    # 2. prepare single-step retrosynthesis model
    RetroProposer = Proposer(args)

    # prepare args for search
    args.beam_size = 5  # Default 5. Must be 1 meaning greedy search or greater or equal 5.
    print(f"beam_size = {args.beam_size}")

    # 3. perform search

    tasks = load_dataset('test')

    # for original routes
    overall_result = np.zeros((args.beam_size, 2))  # 0: results that hit the ground truths
    depth_hit = np.zeros((2, 15, args.beam_size))  # 1: total results
    # for rerank routes
    rerank_overall_result = np.zeros((args.beam_size, 2))  # 0: results that hit the ground truths
    rerank_depth_hit = np.zeros((2, 15, args.beam_size))  # 1: total results

    coverage_rate = np.zeros(6)

    # 存成 json文件，避免重复推理
    ori_route_file = f'checkpoints/{args.save_name}/astar_search_ori_routes.json'       # {args.ckpt_list[0]}-{args.beam_strategy}-
    rerank_route_file = f'checkpoints/{args.save_name}/astar_search_reranked_routes.json'

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

        compare_smis_file = '(astar)_compare_canolize_smis.txt'
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
            final_answer_set = ori_route_dicts[str(epoch)]
            original_answer_set = rerank_route_dicts[str(epoch)]
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




