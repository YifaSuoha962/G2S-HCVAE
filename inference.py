import os
import sys
import os.path as osp
import time

sys.path.insert(0, osp.dirname(os.path.abspath(__file__)))

from rdkit import Chem
import numpy as np
import torch

from models.graph_rel_transformers import Graph_Transformer_Base
from models.graph_vae_transformers import Graph_Transformer_VAE
from models.seq_rel_transformers import Seq_Transformer_Base
from models.seq_vae_transformers import Seq_Transformer_VAE
from models.Graph_LWLV_SMILES import Graph_lv_Transformer_Base
from models.SMILES_LWLV_SMILES import Seq_lv_Transformer_Base
from models.Graph_Lv_Rxn_SMILES import Graph_lv_rxn_Transformer_Base


from models.module_utils import Model_Save, eval_plot, coverage_plot, beam_result_process, process_multi_rxn_coverage
from utils.preprocess_smi import canonicalize_smiles
from utils.parsing import get_parser, post_setting_args
from utils.chem_tools import NODE_FDIM, BOND_FDIM
from utils.wrap_single_smi import InferBatch, InferBatch_wo_rxns

from collections import defaultdict
import pandas as pd

def BiG2S_Inference(args, input_smi, rxn_type):
    Infer_wrapper = InferBatch_wo_rxns('./preprocessed', args)
    batch_graph_input = Infer_wrapper.preprocess(input_smi, rxn_type)
    if args.use_reaction_type:
        dec_cls = 2
    else:
        dec_cls = 1

    args.beam_size = 30
    args.return_num = 30
    ckpt_dir = os.path.join('checkpoints', args.save_name)
    token_idx = Infer_wrapper.token_index
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
            token_idx=Infer_wrapper.token_index,
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
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'S2S_HCVAE':
        predict_module = Seq_Transformer_VAE(
            f_vocab=len(token_idx),
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
    else:  # pure transformer : S2S
        predict_module = Seq_Transformer_Base(
            f_vocab=len(token_idx),
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )

    _, predict_module.model, _, _ = module_saver.load(args.ckpt_list[0], predict_module.model)

    # ckpt_file = osp.join(pretrained_path, 'swa9.ckpt')
    # module_ckpt = torch.load(ckpt_file, map_location=args.device)
    # predict_module.model.load_state_dict(module_ckpt['module_param'])

    with torch.no_grad():
        batch_data = batch_graph_input.to(args.device)
        # , rxn_types, prob_per_token
        predict_result, predict_scores, predict_rxn_types = predict_module.model_predict(
            data=batch_data,
            args=args,
            beam_size=args.beam_size
        )       # TODO: add rxn_type in prediction

    smi_nodes_sorted, prob_nodes_sorted = Infer_wrapper.post_process(predict_result, predict_scores)

    # print(f"pred_smis = \n{smi_nodes_sorted}")

    return smi_nodes_sorted, predict_scores, predict_rxn_types # , prob_per_token


if __name__ == '__main__':

    parser = get_parser(mode='test')
    args = post_setting_args(parser)
    args.use_reaction_type = False
    args.beam_size = 10

    # 需要手动更改的地方： 1. dataset_name, 2. chekpoint_path, 3. 产物分子
    # 用哪个数据集的 checkpoint 和表，就写成哪个

    # 调整 beam search 里的 温度系数
    args.T = 1.2
    assert args.dataset_name in ['uspto_50k', 'pistachio', 'uspto_diverse', 'uspto_50k_infer', 'uspto_full_ms']
    assert args.T in [0.7, 1.2, 1.6]

    if args.use_subs and args.use_reaction_type:
        dec_cls = 2
    elif args.use_subs or args.use_reaction_type:
        dec_cls = 1
    else:
        dec_cls = 0

    #***************************#
    # input_smi = 'N # C c 1 n n ( - c 2 c ( C l ) c c ( C ( F ) ( F ) F ) c c 2 C l ) c c 1 C ( B r ) = C ( C l ) B r'.replace(' ', '')  # demo_a
    # input_smi = 'O C ( c 1 c c c c c 1 ) ( c 1 c c c c c 1 ) c 1 c c c c c 1 Cl'.replace(' ', '')  # demo of pistachio
    # input_smi = 'C c 1 n c ( C # N ) c c c 1 Br'.replace(' ', '')  # demo of pistachio
    # input_smi = "C N c 1 c c c c ( N ) c 1 C # N".replace(' ', '')


    # input_smi = "Nc1ncc(C(F)(F)F)cc1Cl"
    input_smi = "Nc1ccc(-c2ccccc2)cn1"
    refs = ['F C ( F ) ( F ) c 1 c n c ( Cl ) c ( Cl ) c 1 . N'.replace(' ', ''),
            'F C ( F ) ( F ) c 1 c c c ( N Cl ) n c 1 . N c 1 c c c ( C ( F ) ( F ) F ) c n 1'.replace(' ', ''),
            'N c 1 c c c ( C ( F ) ( F ) F ) c n 1 . O = c 1 n ( Cl ) c ( = O ) n ( Cl ) c ( = O ) n 1 Cl'.replace(' ', ''),
            'N c 1 c c c ( C ( F ) ( F ) F ) c n 1 . O = C 1 c 2 c c c c c 2 C ( = O ) N 1 Cl'.replace(' ', '')]

    # 为了方便，用数字代表反应类型序号
    rxn_type = 0
    #***************************#

    start = time.perf_counter()
    # , token_score
    top_k, tot_score, rxn_types = BiG2S_Inference(args, input_smi, rxn_type)
    filtered_top_k = [smi.split(',')[1] for smi in top_k[0]]
    max_len = max([len(s) for s in filtered_top_k])

    # print(f'infer res = {filtered_top_k}')

    end = time.perf_counter()
    # print(top_k)
    print(tot_score)
    print('推理时间: %s 秒' % (end - start))

    # match_list = [0] * 30
    # valid_list = [0] * 30
    # for i in range(len(filtered_top_k)):
    #     res_smi = filtered_top_k[i]
    #     if not canonicalize_smiles(res_smi) == '':
    #         valid_list[i] = 1
    #         if res_smi in refs:
    #             match_list[i] = 1

    # filtered_top_k = ['FC(F)(F)c1cnc(Cl)c(Cl)c1.N', 'FC(F)(F)c1cnc(Cl)c(Cl)c1.[NH4+]', 'Nc1ccc(C(F)(F)F)cn1.O=P(Cl)(Cl)Cl',
    #              'Nc1ccc(C(F)(F)F)cn1.O=C1c2ccccc2C(=O)N1Cl', 'CC(=O)Nc1ccc(C(F)(F)F)cn1.Cl',
    #              'Nc1ccc(C(F)(F)F)cn1.[O-][I+2]([O-])[O-]',
    #              'ClC(Cl)Cl.Nc1ccc(C(F)(F)F)cn1', 'CC(=O)Nc1ccc(C(F)(F)F)cn1.Cl[Sn]Cl',
    #              'CC(C)(C)C(=O)Nc1ccc(C(F)(F)F)cn1',
    #              'CC(=O)Nc1ccc(C(F)(F)F)cn1']
    len_res = len(filtered_top_k)

    valid_list = [0] * len_res
    match_list = [0] * len_res
    for i in range(len_res):
        res_smi = filtered_top_k[i]
        if not canonicalize_smiles(res_smi) == '':
            valid_list[i] = 1
            if res_smi in refs:
                match_list[i] = 1


    infer_df = pd.DataFrame({'pred_reacts': filtered_top_k, 'valid': valid_list, 'match': match_list})
    if not os.path.exists('infer_demo'):
        os.makedirs('infer_demo')
    infer_df.to_csv('infer_demo/infer_result.csv', index=False)


    known_res = ['FC(F)(F)c1cnc(Cl)c(Cl)c1.N', 'FC(F)(F)c1cnc(Cl)c(Cl)c1.[NH4+]', 'Nc1ccc(C(F)(F)F)cn1.O=P(Cl)(Cl)Cl',
                 'CC(=O)Nc1ncc(C(F)(F)F)cc1Cl', 'CC(=O)Nc1ccc(C(F)(F)F)cn1.Cl', 'Nc1ccc(C(F)(F)F)cn1.[O-][I+2]([O-])[O-]',
                 'ClC(Cl)Cl.Nc1ccc(C(F)(F)F)cn1', 'CC(=O)Nc1ccc(C(F)(F)F)cn1.Cl[Sn]Cl',  'CC(C)(C)C(=O)Nc1ccc(C(F)(F)F)cn1',
                 'CC(=O)Nc1ccc(C(F)(F)F)cn1']

    print(f"res = {refs}")