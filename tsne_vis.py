import torch
import torch.nn as nn
import logging
import argparse
import random

import os
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.std import trange

# retrog2s utils
proj_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_dir, 'preprocessed')
origin_dir = os.path.join(proj_dir, 'data')
from utils.parsing import get_parser, post_setting_args
from rerank import Proposer     # , get_rerank_scores
from utils.data_loader import G2SFullDataset, S2SFullDataset


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


def visualize_embeddings(args, mode,
                         embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         n_components: int = 2,
                         random_state: int = 42):
    """
    可视化高维嵌入向量

    参数:
        embeddings: 形状为 [batch_size, hidden_size] 的PyTorch张量
        labels: 形状为 [batch_size] 的标签张量
        n_components: 降维后的维度 (2或3)
        title: 图像标题
        random_state: 随机种子
        figsize: 图像大小
    """
    # 转换数据到CPU和numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # t-SNE降维
    # tsne = TSNE(n_components=n_components,
    #             random_state=random_state,
    #             perplexity=20,          # 小于默认值30，增强局部结构
    #             learning_rate=500,      # 增大步长强化分离
    #             early_exaggeration=24,  # 加大早期放大因子
    #             n_iter=1000,            # 增加迭代次数
    #             init='pca')

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=20,  # 减小值增强局部紧凑性 (建议范围5-15)
        learning_rate=200,  # 减小步长使收敛更稳定 (建议范围100-300)
        early_exaggeration=8,  # 减小早期放大因子 (建议范围8-16)
        n_iter=500,  # 增加迭代次数确保充分收敛
        init='pca',  # 保持PCA初始化
        metric='cosine',
        angle=0.3  # 减小角度提高精度(0.2-0.5)
    )

    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # 可视化
    plt.figure(figsize=(8, 6))

    # 离散化表示标签
    unique_labels = np.unique(labels_np)
    num_classes = len(unique_labels)

    # 为每个类别创建自定义颜色映射
    cmap = plt.get_cmap('viridis', num_classes)

    if n_components == 2:
        # 为每个类别单独绘制散点
        for label in unique_labels:
            mask = labels_np == label
            plt.scatter(embeddings_tsne[mask, 0],
                        embeddings_tsne[mask, 1],
                        color=cmap(label),
                        alpha=0.5,
                        label=f'{label}',
                        linewidth=0.3)
        # edgecolor = sns.dark_palette(cmap(label), n_colors=1)[0],   # 'w',  # 白色边缘增强区分度
        #
        # 添加离散图例
        plt.legend(title='Latent class $c$',
                   bbox_to_anchor=(0.99, 0.98),  # 右上角内部坐标 (x,y)
                   loc='upper right',  # 锚点定位到右上
                   borderaxespad=0.4,  # 增加边框间距
                   frameon=True,
                   fontsize=12,
                   title_fontsize=14,
                   ncol=2,
                   handletextpad=0.4,
                   columnspacing=0.7)

        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度

    elif n_components == 3:
        ax = plt.axes(projection='3d')
        for label in unique_labels:
            mask = labels_np == label
            ax.scatter3D(embeddings_tsne[mask, 0],
                         embeddings_tsne[mask, 1],
                         embeddings_tsne[mask, 2],
                         color=cmap(label),
                         alpha=0.4,
                         label=f'{label}',
                         edgecolor='w',
                         linewidth=0.3)

        ax.legend(title='Latent class $c$',
                  bbox_to_anchor=(1.1, 0.9),
                  loc='upper left',
                  fontsize=12,
                  title_fontsize=14,
                  ncol=2,  # 2列布局
                  handletextpad=0.5,  # 调整文本与图例标记的间距
                  columnspacing=0.8)

        ax.set_xticks([])  # 移除3D图的x轴刻度
        ax.set_yticks([])  # 移除3D图的y轴刻度
        ax.set_zticks([])  # 移除3D图的z轴刻度
    else:
        raise ValueError("n_components 只能是2或3")

    # 添加图例和标题
    # plt.colorbar(scatter, label='Class Label')
    # plt.title(f'{args.dataset_name}')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')

    # 显示图形
    save_dir = 'tsne-plot'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{args.dataset_name}-{mode}-{n_components}dim.svg')
    plt.show()
    save_path = os.path.join(save_dir, f'{args.dataset_name}-{mode}-{n_components}dim.pdf')
    plt.savefig(save_path)



def tsne_procedure(dataset, proposer):
    pass



if __name__ == "__main__":
    parser = get_parser(mode = 'test')
    parser.add_argument('--data_mode', type=str, choices=['train', 'eval'], default='eval')
    parser.add_argument('--tsne_dim', type=int, choices=[2, 3], default=2)

    args = post_setting_args(parser)

    RetroProposer = Proposer(args)
    if args.data_mode == 'train':
        if args.representation_form == 'graph2smiles':
            tsne_dataset = G2SFullDataset(
                origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode='train',
                dist_block=args.graph_dist_block,
                task=args.train_task
            )
        else:
            tsne_dataset = S2SFullDataset(
                origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode='train',
                dist_block=args.graph_dist_block,
                task=args.train_task)
    else:
        # args.data_mode == 'eval':
        if args.representation_form == 'graph2smiles':
            tsne_dataset = G2SFullDataset(
                origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode=args.mode,
                dist_block=args.graph_dist_block,
                task=args.eval_task,
                use_split=args.use_splited_data,
                split_data_name=args.split_data_name
            )
        else:
            tsne_dataset = S2SFullDataset(
                origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode=args.mode,
                dist_block=args.graph_dist_block,
                task=args.eval_task,
                use_split=args.use_splited_data,
                split_data_name=args.split_data_name
            )

    rxn_pairs = []
    rxn_types = []

    with (torch.no_grad()):
        tsne_dataset.get_batch()
        data_loader = DataLoader(
            dataset=tsne_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )
        teval = trange(tsne_dataset.batch_step)
        for step, batch in zip(teval, data_loader):
            batch = batch.to(args.device)

            tmp_rxn_pair, tmp_rxn_type = RetroProposer.model.collect_embeddings(
                data=batch
            )

            # 确保输出是张量并存储
            if not isinstance(tmp_rxn_pair, torch.Tensor):
                tmp_rxn_pair = torch.tensor(tmp_rxn_pair, device='cpu')
            if not isinstance(tmp_rxn_type, torch.Tensor):
                tmp_rxn_type = torch.tensor(tmp_rxn_type, device='cpu')


            rxn_pairs.append(tmp_rxn_pair.cpu())  # 转移到CPU避免显存溢出
            rxn_types.append(tmp_rxn_type.cpu())

    print(f"len(rxn_pairs) = {len(rxn_pairs)}")
    # 合并所有批次的输出
    if len(rxn_pairs) > 0:
        rxn_pairs = torch.cat(rxn_pairs, dim=0)  # 最终形状 [n*batch_size, hidden_size]
        rxn_types = torch.cat(rxn_types, dim=0)  # 最终形状 [n*batch_size]
    else:
        rxn_pairs = torch.empty((0, 2))  # 空张量处理
        rxn_types = torch.empty((0,))

    visualize_embeddings(args, args.data_mode, rxn_pairs, rxn_types, n_components=args.tsne_dim)


