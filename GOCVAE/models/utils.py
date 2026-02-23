import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import sys, os
from tqdm import tqdm
import pickle

from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from collections import defaultdict


def normalize_hvg(adata, target_sum=1e4, size_factors=True, scale_input=True, logtrans_input=True,
                  n_top_genes=2000, normalize_input=True):

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_total(adata, target_sum=target_sum, key_added='size_factors')
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if scale_input:
        sc.pp.scale(adata)

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    return adata


def train_test_split(adata, train_frac=0.85):
    """
        Split ``adata`` into train and test annotated datasets.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        train_frac: float
            Fraction of observations (cells) to be used in training dataset. Has to be a value between 0 and 1.

        Returns
        -------
        train_adata: :class:`~anndata.AnnData`
            Training annotated dataset.
        valid_adata: :class:`~anndata.AnnData`
            Test annotated dataset.
    """
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def label_encoder(adata, le=None, condition_key='condition'):
    """
        Encode labels of Annotated `adata` matrix using sklearn.preprocessing.LabelEncoder class.
        对输入的 AnnData 对象中的条件（扰动条件）进行标签编码。它使用一个字典将条件标签（如字符串或类别）转换为数值编码。
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        Example
        --------
        >>> import trvae
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = label_encoder(train_data)
    """
    if le is not None:
        assert isinstance(le, dict)

    unique_conditions = np.unique(adata.obs[condition_key]).tolist()
    if le is None:
        le = dict()
        for idx, condition in enumerate(unique_conditions):
            le[condition] = idx

    assert set(unique_conditions).issubset(list(le.keys()))
    labels = np.zeros(adata.shape[0])
    for condition, label in le.items():
        labels[adata.obs[condition_key] == condition] = label

    return labels.reshape(-1, 1), le


def create_dictionary(conditions, target_conditions=[]):
    if isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary


"""
GO网络生成相关函数（仿照GEARS实现）
"""
def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    获取基因本体论边列表

    - 计算单个基因与其它所有基因之间的 GO 相似度（Jaccard），
    - 返回相似度大于0.1的基因对及其分数（用于GO网络边的生成）。

    输入:
      args: (g1, gene2go)
        g1: 当前基因名（字符串）
        gene2go: 字典，key为基因名，value为该基因注释的GO term集合（set）
    输出:
      edge_list: 列表，元素为三元组 (g1, g2, score)
        其中 score 为 g1 与 g2 的 GO term Jaccard 相似度，score > 0.1
    """
    g1, gene2go = args
    edge_list = []
    # 遍历所有目标基因，与 g1 计算 GO term 的 Jaccard 相似度
    for g2 in gene2go.keys():
        # Jaccard 相似度 = 交集/并集
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(gene2go[g1].union(gene2go[g2]))
        # 只保留相似度大于 0.1 的边（稀疏处理，过滤弱相关）
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list


def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    基于GO构建的基因-基因网络

    输入：
      data_path: 数据目录
      pert_list: 需要构建GO图的基因列表（通常为可扰动基因）
      data_name: 数据集名（用于输出文件命名）
      num_workers: 多进程并发数
      save: 是否保存生成的边表到本地
    输出：
      DataFrame, 三列 [source, target, importance]：基因对及其GO相似度
    """

    fname = os.path.join(data_path, f'go_essential_{data_name}.csv')
    # 如果之前已经生成过，直接读取（避免重复计算）
    if os.path.exists(fname):
        return pd.read_csv(fname)

    # 读取预先生成好的 gene2go 字典（基因名 → GO term 列表）
    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    # 只保留 pert_list（可扰动基因）对应的 GO 注释
    gene2go = {i: gene2go[i] for i in pert_list if i in gene2go}

    # 多进程遍历 pert_list 内每个基因，与其它所有基因两两比较GO集合（调用 get_GO_edge_list）
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                 total=len(gene2go.keys())))
    # 聚合所有基因对的边信息
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    # 转为 DataFrame 并重命名列名：source/target/importance（基因1/基因2/相似度分数）
    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})

    # 可选：保存到本地文件
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list


def get_similarity_network(k, data_path, data_name, pert_list=None):
    """
        构建GO功能相似性基因-基因网络图。
        只支持本地动态构建，输入为基因列表pert_list，输出DataFrame三列[source, target, importance]。

        参数:
          network_type: 'co-express'（共表达图）或 'go'（GO功能图）
          adata: AnnData对象，存储单细胞表达矩阵
          threshold, k: 边的筛选参数（阈值或保留前k个邻居）
          data_path, data_name: 路径及数据名
          split, seed, train_gene_set_size, set2conditions: 训练集相关参数
          default_pert_graph: 是否使用官方预制GO图（True: 用官方图，False: 动态构建）
          pert_list: 如果动态构建GO图，需要指定参与扰动的基因列表
        返回:
          df_out: DataFrame, 三列 [source, target, importance]，可用于后续建图
    """

    # make_GO 负责遍历 pert_list 内所有基因，两两计算GO term的Jaccard相似度
    df_jaccard = make_GO(data_path, pert_list, data_name)

    # 保留每个目标基因最相似的(k+1)个邻居（top-k稀疏处理）
    df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1, ['importance'])).reset_index(drop=True)

    return df_out


def validate_go_graph(go_graph, gene_list):
    """
    检验GO图生成结果的质量

    参数:
        go_graph: DataFrame, make_GO()的输出结果
        gene_list: list, 原始基因列表

    返回:
        dict: 验证统计信息
    """
    print("=== GO图质量检验 ===")

    # 基本统计
    n_edges = len(go_graph)
    genes_in_graph = set(go_graph['source']) | set(go_graph['target'])
    n_genes_in_graph = len(genes_in_graph)
    n_genes_total = len(gene_list)

    # 基因覆盖率
    coverage = len(genes_in_graph & set(gene_list)) / n_genes_total

    # 权重统计
    weight_stats = go_graph['importance'].describe()

    # 度分布统计
    degree_dist = go_graph['source'].value_counts()

    stats = {
        'n_edges': n_edges,
        'n_genes_in_graph': n_genes_in_graph,
        'n_genes_total': n_genes_total,
        'coverage': coverage,
        'avg_weight': weight_stats['mean'],
        'weight_std': weight_stats['std'],
        'min_weight': weight_stats['min'],
        'max_weight': weight_stats['max'],
        'avg_degree': degree_dist.mean(),
        'max_degree': degree_dist.max()
    }

    # 打印结果
    print(f"边数: {n_edges}")
    print(f"图中基因数: {n_genes_in_graph}")
    print(f"总基因数: {n_genes_total}")
    print(f"基因覆盖率: {coverage:.3f}")
    print(f"平均边权重: {stats['avg_weight']:.4f}")
    print(f"权重范围: [{stats['min_weight']:.4f}, {stats['max_weight']:.4f}]")
    print(f"平均节点度数: {stats['avg_degree']:.1f}")
    print(f"最大节点度数: {stats['max_degree']}")

    # 质量评估
    if coverage < 0.3:
        print("警告: 基因覆盖率过低，可能需要检查gene2go_all.pkl")
    elif coverage < 0.6:
        print("注意: 基因覆盖率较低，部分基因可能缺少GO注释")
    else:
        print("基因覆盖率良好")

    if stats['avg_weight'] < 0.15:
        print("⚠注意: 平均权重较低，基因间GO相似性可能较弱")
    else:
        print("边权重分布合理")

    return stats


def save_go_graph_for_pawine(go_graph, gene_list, output_path):
    """
    将GO图转换为PaWine训练需要的格式并保存

    参数:
        go_graph: DataFrame, make_GO()的输出结果
        gene_list: list, 基因列表（用于确保基因顺序一致）
        output_path: str, 输出文件路径

    PaWine需要的格式:
        - 文本文件，每行格式: source_gene target_gene weight
        - 基因名需要与后续模型训练时的基因顺序一致
    """
    print("=== 转换为PaWine格式 ===")

    # 检查基因一致性
    genes_in_graph = set(go_graph['source']) | set(go_graph['target'])
    genes_in_list = set(gene_list)

    missing_genes = genes_in_graph - genes_in_list
    if missing_genes:
        print(f"警告: GO图中有 {len(missing_genes)} 个基因不在基因列表中")
        print(f"示例缺失基因: {list(missing_genes)[:5]}")

    # 过滤掉不在基因列表中的边
    filtered_graph = go_graph[
        go_graph['source'].isin(gene_list) &
        go_graph['target'].isin(gene_list)
        ].copy()

    print(f"过滤后边数: {len(filtered_graph)} (原始: {len(go_graph)})")

    # 统计顶点数和边数
    vertices_in_edges = set(filtered_graph['source']) | set(filtered_graph['target'])
    num_vertices = len(vertices_in_edges)
    num_edges = len(filtered_graph)

    # 保存为PaWine格式（空格分隔，无表头）
    with open(output_path, 'w') as f:
        # 后续行：源节点 目标节点 权重
        for _, row in filtered_graph.iterrows():
            f.write(f"{row['source']} {row['target']} {row['importance']:.6f}\n")

    print(f"PaWine格式GO图已保存: {output_path}")
    print(f"格式: 第一行 {num_vertices} {num_edges}, 后续 {num_edges} 行边数据")

    # 同时保存基因顺序映射（用于后续集成）
    gene2idx = {gene: idx for idx, gene in enumerate(gene_list)}
    mapping_path = output_path.replace('.txt', '_gene2idx.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(gene2idx, f)
    print(f"基因索引映射已保存: {mapping_path}")

    return filtered_graph


def build_and_validate_go_graph(gene_list, data_path='./data', data_name='default',
                                k=20, output_dir='./data'):
    """
    完整的GO图构建和验证流程

    参数:
        gene_list: list, 基因名列表
        data_path: str, gene2go_all.pkl所在路径
        data_name: str, 数据集名称
        k: int, 每个基因保留的邻居数
        output_dir: str, 输出目录

    返回:
        go_graph: DataFrame, 构建的GO图
        stats: dict, 验证统计信息
    """
    print(f"=== 开始构建GO图 ===")
    print(f"基因数: {len(gene_list)}")
    print(f"数据集: {data_name}")
    print(f"K值: {k}")

    # 1. 构建GO图
    go_graph = get_similarity_network(
        k=k,
        data_path=data_path,
        data_name=data_name,
        pert_list=gene_list
    )

    # 2. 验证质量
    stats = validate_go_graph(go_graph, gene_list)

    # 3. 保存标准CSV格式
    csv_path = os.path.join(output_dir, f'go_graph_{data_name}.csv')
    go_graph.to_csv(csv_path, index=False)
    print(f"标准格式GO图已保存: {csv_path}")

    # 4. 保存PaWine格式
    pawine_path = os.path.join(output_dir, f'go_graph_{data_name}_pawine.txt')
    filtered_graph = save_go_graph_for_pawine(go_graph, gene_list, pawine_path)

    print(f"=== GO图构建完成 ===")
    return go_graph, stats


"""
基因ID转换函数（简化版）
"""
def convert_ensembl_to_symbol(ensembl_ids):
    """
    将Ensembl ID转换为Gene Symbol
    使用mygene包，失败则跳过该基因

    参数:
        ensembl_ids: list, Ensembl ID列表

    返回:
        list: 转换后的Gene Symbol列表（去除转换失败的）
    """
    try:
        import mygene
        mg = mygene.MyGeneInfo()

        print(f"正在转换 {len(ensembl_ids)} 个基因ID...")
        results = mg.querymany(
            ensembl_ids,
            scopes='ensembl.gene',
            fields='symbol',
            species='human'     # 当使用haber数据库时应该修改这部分换成老鼠
        )

        converted_genes = []
        for result in results:
            if 'symbol' in result and result['symbol']:
                converted_genes.append(result['symbol'])

        print(f"成功转换 {len(converted_genes)} 个基因")
        return converted_genes

    except ImportError:
        print("❌ 需要安装mygene包: pip install mygene")
        return []
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return []


def filter_genes_with_go_annotation(gene_list, data_path):
    """
    过滤掉没有GO注释的基因

    参数:
        gene_list: list, 基因符号列表
        data_path: str, 数据路径

    返回:
        list: 有GO注释的基因列表
    """
    gene2go_file = os.path.join(data_path, 'gene2go_all.pkl')

    with open(gene2go_file, 'rb') as f:
        gene2go = pickle.load(f)

    available_genes = set(gene2go.keys())
    filtered_genes = [gene for gene in gene_list if gene in available_genes]

    print(f"过滤结果: {len(filtered_genes)}/{len(gene_list)} 个基因有GO注释")

    return filtered_genes


"""
对生成的GO图使用PaWine，得到优化后的基因嵌入表
"""
def run_pawine_training(go_network_file, gene_list, embedding_dim=128, output_dir='./'):
    """
    运行PaWine训练，输出numpy格式的基因嵌入

    参数:
        go_network_file: str, GO网络文件路径 (go_graph_*_pawine.txt)
        gene_list: list, 基因名称列表
        embedding_dim: int, 嵌入维度
        output_dir: str, 输出目录

    返回:
        str: numpy格式基因嵌入文件路径
    """
    import subprocess
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # 1. 编译PaWine
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'models', 'pawine.cpp'),  # trvae/models/pawine.cpp
        os.path.join(os.path.dirname(__file__), '..', 'models', 'pawine.cpp'),  # 保留原来的作为备选
        os.path.join(os.getcwd(), 'models', 'pawine.cpp'),
        './pawine.cpp'
    ]

    pawine_cpp_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pawine_cpp_path = path
            break

    if pawine_cpp_path is None:
        raise FileNotFoundError("❌ 找不到pawine.cpp文件")

    pawine_exe = os.path.join(output_dir, 'pawine')

    if not os.path.exists(pawine_exe):
        cmd = f"g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result {pawine_cpp_path} -o {pawine_exe} -lgsl -lm -lgslcblas"
        subprocess.run(cmd, shell=True, check=True)
        print(f"PaWine compiled: {pawine_exe}")

    # 2. 从GO网络文件中提取实际的基因顺序
    actual_genes = set()
    with open(go_network_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                actual_genes.add(parts[0])
                actual_genes.add(parts[1])

    # 按字典序排序（PaWine内部可能是这样排序的）
    n_actual_genes = len(actual_genes)
    print(f"GO网络中实际基因数: {n_actual_genes}")
    print(f"输入gene_list基因数: {len(gene_list)}")

    # 3. 运行PaWine
    embeddings_txt = os.path.join(output_dir, 'embeddings.txt')
    pawine_exe_abs = os.path.abspath(pawine_exe)  # 可执行文件的绝对路径
    go_network_abs = os.path.abspath(go_network_file)  # 网络文件的绝对路径

    # 切换到工作目录运行
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        cmd = f"{pawine_exe_abs} -train {go_network_abs} -output embeddings.txt -size {embedding_dim} -negative 1 -samples 100 -step 0.01 -lambda 0.01 -lambda_d 0.01 -threads 4 -negchoice 0"
        print(f"Running PaWine training...")
        subprocess.run(cmd, shell=True, check=True)
        print(f" PaWine training completed")
    finally:
        os.chdir(original_dir)

    # 4. 转换为numpy格式
    embeddings_dict = {}
    with open(embeddings_txt, 'r') as f:
        lines = f.readlines()[1:]  # 跳过第一行元数据

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            gene_name = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings_dict[gene_name] = embedding

    # 按gene_list顺序排列
    embedding_matrix = np.zeros((len(gene_list), embedding_dim))
    for i, gene in enumerate(gene_list):
        if gene in embeddings_dict:
            embedding_matrix[i] = embeddings_dict[gene]

    # 保存为numpy格式
    numpy_file = os.path.join(output_dir, 'gene_embeddings.npy')
    np.save(numpy_file, embedding_matrix)

    # 清理临时文件
    if os.path.exists(embeddings_txt):
        os.remove(embeddings_txt)

    print(f"Gene embeddings saved: {numpy_file}")
    print(f"Shape: {embedding_matrix.shape}")

    return numpy_file


def prepare_gene_embeddings_for_model(adata, condition_key='condition',
                                    data_path='../data/GEARS/',
                                    go_network_file=None,
                                    output_dir='../data/GEARS/test_pawine/',
                                    embedding_dim=32):
    """
    为模型准备基因嵌入的一站式函数

    参数:
        adata: anndata对象
        condition_key: 条件键名

    返回:
        dict: gene_embedding_config字典，可直接传给EnhancedCLDRCVAE
    """
    # 提取基因和条件
    from trvae.utils import convert_ensembl_to_symbol, filter_genes_with_go_annotation

    ensembl_ids = adata.var_names.tolist()
    converted_genes = convert_ensembl_to_symbol(ensembl_ids)
    gene_list = filter_genes_with_go_annotation(converted_genes, data_path)

    # 如果没有指定GO网络文件，使用默认路径
    if go_network_file is None:
        go_network_file = os.path.join(data_path, 'GEARS', 'GO', 'go_graph_sciplex_test_pawine.txt')

    # 运行完整流程
    embedding_path = run_pawine_training(
        go_network_file=go_network_file,
        gene_list=gene_list,
        embedding_dim=embedding_dim,
        output_dir=output_dir
    )

    # 保存基因映射
    gene2idx = {gene: idx for idx, gene in enumerate(gene_list)}
    mapping_path = os.path.join(output_dir, 'gene2idx.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(gene2idx, f)

    return {
        'embedding_path': embedding_path,
        'gene_mapping_path': mapping_path,
        'embedding_dim': embedding_dim,
        'freeze_embeddings': True,
        'combo_mlp_layers': [64, embedding_dim],
        'combo_mlp_dropout': 0.1,
    }


# 预处理构建训练批次
def ensure_required_columns(adata):
    """确保 adata 有必需的列"""

    # 1. 添加 nperts 列
    if 'nperts' not in adata.obs.columns:
        print("缺少 'nperts' 列，正在自动添加...")

        def count_perts(condition):
            if condition == 'ctrl':
                return 0
            # 移除 ctrl 后计数
            clean = condition.replace('+ctrl', '').replace('ctrl+', '')
            return 1 if '+' not in clean else clean.count('+') + 1

        adata.obs['nperts'] = adata.obs['condition'].apply(count_perts)
        print(f"已添加 'nperts' 列")
        print(f"  分布: {dict(adata.obs['nperts'].value_counts())}")

    # 2. 添加 perturbation 列
    if 'perturbation' not in adata.obs.columns:
        print("缺少 'perturbation' 列，正在自动添加...")

        def extract_pert(condition):
            if condition == 'ctrl':
                return 'control'
            return condition.replace('+ctrl', '').replace('ctrl+', '')

        adata.obs['perturbation'] = adata.obs['condition'].apply(extract_pert)
        print(f"✓ 已添加 'perturbation' 列")

    return adata



def create_smart_batches(adata, batch_size=512, data_path='../data/GEARS/', go_similarity_dict=None,
                         single_gene_ratio=0.6, multi_gene_ratio=0.3, control_ratio=0.1,
                         go_similarity_threshold=0.15, min_go_group_size=10,
                         similar_gene_ratio=0.7):
    """
    为PaWine训练创建智能批次分组

    Parameters:
    - adata: 训练数据AnnData对象
    - batch_size: 批次大小
    - go_similarity_dict: 已构建的GO相似性字典，如果提供则直接使用
    - single_gene_ratio: 单基因样本在批次中的比例
    - multi_gene_ratio: 多基因样本比例
    - control_ratio: 控制样本比例
    - go_similarity_threshold: GO相似性阈值
    - min_go_group_size: GO功能组最小大小
    - similar_gene_ratio: 批次内相似基因的比例

    Returns:
    - batch_indices_list: 预生成的批次索引列表
    - go_similarity_dict: GO相似性字典
    """
    adata = ensure_required_columns(adata)

    # 1. 数据分类：按扰动类型分组
    single_gene_indices = []
    multi_gene_indices = []
    control_indices = []

    for idx, (nperts, condition) in enumerate(zip(adata.obs['nperts'], adata.obs['perturbation'])):
        if nperts == 1 and condition != 'control':
            single_gene_indices.append(idx)
        elif nperts > 1:
            multi_gene_indices.append(idx)
        else:  # nperts == 0 or condition == 'control'
            control_indices.append(idx)

    print(f"数据分类完成:")
    print(f"  单基因样本: {len(single_gene_indices)}")
    print(f"  多基因样本: {len(multi_gene_indices)}")
    print(f"  控制样本: {len(control_indices)}")

    # 2. 使用已有的GO网络或构建新的
    if go_similarity_dict is not None:
        print(f"使用已构建的GO网络: {len(go_similarity_dict)} 个节点")
    else:
        print("未提供GO网络，开始构建...")
        if 'symbol' in adata.var.columns:
            all_genes = adata.var['symbol'].unique().tolist()
        elif 'gene_name' in adata.var.columns:
            all_genes = adata.var['gene_name'].unique().tolist()
        else:
            all_genes = adata.var_names.tolist()

        print(f"使用全部var基因构建GO网络，共 {len(all_genes)} 个基因")

        # 使用已有的utils函数
        filtered_genes = filter_genes_with_go_annotation(all_genes, data_path)
        print(f"过滤结果: {len(filtered_genes)}/{len(all_genes)} 个基因有GO注释")

        if len(filtered_genes) < min_go_group_size:
            print(f"警告: GO注释基因过少({len(filtered_genes)})，使用随机分组")
            go_similarity_dict = {}
        else:
            # 构建GO相似性网络
            go_df = get_similarity_network(k=20, data_path=data_path,
                                           data_name='norman2019_full', pert_list=filtered_genes)

            # 过滤低相似度边
            go_df_filtered = go_df[go_df['importance'] >= go_similarity_threshold]
            print(f"GO网络边数: {len(go_df)} -> {len(go_df_filtered)} (阈值: {go_similarity_threshold})")

            # 构建相似性字典
            go_similarity_dict = {}
            for _, row in go_df_filtered.iterrows():
                source, target, weight = row['source'], row['target'], row['importance']
                if source not in go_similarity_dict:
                    go_similarity_dict[source] = []
                go_similarity_dict[source].append((target, weight))
                # 双向关系
                if target not in go_similarity_dict:
                    go_similarity_dict[target] = []
                go_similarity_dict[target].append((source, weight))

            # 对相似基因按权重排序
            for gene in go_similarity_dict:
                go_similarity_dict[gene].sort(key=lambda x: x[1], reverse=True)

    # 3. 基因功能聚类（移到if-else外部，确保总是被执行）
    single_gene_data = adata[single_gene_indices]
    unique_perturbed_genes = single_gene_data.obs['perturbation'].unique()

    if len(go_similarity_dict) > 0:
        go_functional_groups = _cluster_genes_by_go_similarity(
            unique_perturbed_genes,  # 只聚类扰动基因
            go_similarity_dict,  # 但使用全部基因的GO网络
            min_go_group_size
        )
    else:
        # 如果GO图为空，使用随机分组
        go_functional_groups = {'random': list(unique_perturbed_genes)}

    print(f"GO功能分组: {len(go_functional_groups)} 个组")

    # 4. 构建基因到样本的映射
    gene_to_sample_indices = defaultdict(list)
    for idx in single_gene_indices:
        gene = adata.obs['perturbation'].iloc[idx]
        gene_to_sample_indices[gene].append(idx)

    # 5. 生成智能批次
    batch_indices_list = []

    # 计算每个批次中各类样本的数量
    n_single_per_batch = int(batch_size * single_gene_ratio)
    n_multi_per_batch = int(batch_size * multi_gene_ratio)
    n_control_per_batch = batch_size - n_single_per_batch - n_multi_per_batch

    # 计算相似和不相似基因的数量
    n_similar_genes = int(n_single_per_batch * similar_gene_ratio)
    n_different_genes = n_single_per_batch - n_similar_genes

    print(f"每批次构成: 单基因{n_single_per_batch}({n_similar_genes}相似+{n_different_genes}不相似), "
          f"多基因{n_multi_per_batch}, 控制{n_control_per_batch}")

    # 为每个GO功能组生成批次
    total_batches_needed = max(1, len(single_gene_indices) // n_single_per_batch)

    for batch_idx in range(total_batches_needed):
        current_batch = []

        # 选择主导的GO功能组
        main_group_id = batch_idx % len(go_functional_groups)
        main_group_genes = list(go_functional_groups.values())[main_group_id]

        # 添加相似基因样本
        similar_samples = _sample_from_gene_group(main_group_genes, gene_to_sample_indices,
                                                  n_similar_genes)
        current_batch.extend(similar_samples)

        # 添加不相似基因样本
        other_groups = [genes for i, genes in enumerate(go_functional_groups.values())
                        if i != main_group_id]
        if other_groups:
            other_genes = [gene for group in other_groups for gene in group]
            different_samples = _sample_from_gene_group(other_genes, gene_to_sample_indices,
                                                        n_different_genes)
            current_batch.extend(different_samples)

        # 添加多基因样本
        if len(multi_gene_indices) >= n_multi_per_batch:
            multi_samples = np.random.choice(multi_gene_indices, n_multi_per_batch, replace=False)
            current_batch.extend(multi_samples)
        else:
            current_batch.extend(multi_gene_indices)

        # 添加控制样本
        remaining_needed = batch_size - len(current_batch)
        if len(control_indices) >= remaining_needed and remaining_needed > 0:
            control_samples = np.random.choice(control_indices, remaining_needed, replace=False)
            current_batch.extend(control_samples)

        # 确保批次大小并添加到列表
        current_batch = current_batch[:batch_size]
        if len(current_batch) >= batch_size * 0.8:  # 至少80%满才使用
            batch_indices_list.append(np.array(current_batch))

    print(f"智能批次构建完成: {len(batch_indices_list)} 个批次")

    return batch_indices_list, go_similarity_dict


def _cluster_genes_by_go_similarity(genes, go_similarity_dict, min_group_size):
    """
    基于GO相似性对基因进行聚类分组
    """
    if len(genes) < min_group_size:
        return {'single_group': genes}

    # 构建相似性矩阵
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    n_genes = len(genes)
    similarity_matrix = np.zeros((n_genes, n_genes))

    for i, gene1 in enumerate(genes):
        similarity_matrix[i, i] = 1.0
        if gene1 in go_similarity_dict:
            for gene2, weight in go_similarity_dict[gene1]:
                if gene2 in gene_to_idx:
                    j = gene_to_idx[gene2]
                    similarity_matrix[i, j] = weight
                    similarity_matrix[j, i] = weight

    # 聚类
    try:
        distance_matrix = 1 - similarity_matrix
        n_clusters = max(2, min(8, n_genes // min_group_size))

        mds = MDS(n_components=min(10, n_genes - 1), dissimilarity='precomputed', random_state=42)
        gene_embeddings = mds.fit_transform(distance_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(gene_embeddings)

        # 构建分组
        groups = defaultdict(list)
        for gene, cluster_id in zip(genes, cluster_labels):
            groups[f'group_{cluster_id}'].append(gene)

        # 过滤太小的组
        filtered_groups = {gid: genes_list for gid, genes_list in groups.items()
                           if len(genes_list) >= min_group_size}

        if not filtered_groups:
            return {'single_group': genes}

        return filtered_groups

    except Exception as e:
        print(f"聚类失败，使用单一分组: {e}")
        return {'single_group': genes}


def _sample_from_gene_group(genes, gene_to_sample_indices, n_samples):
    """
    从基因组中采样指定数量的样本
    """
    available_samples = []
    for gene in genes:
        available_samples.extend(gene_to_sample_indices.get(gene, []))

    if len(available_samples) == 0:
        return []

    n_to_sample = min(n_samples, len(available_samples))
    return np.random.choice(available_samples, n_to_sample, replace=False).tolist()


"""
GEARS - 扰动条件解析相关函数
"""
class GeneSimNetwork:
    """
    Gene Similarity Network
    将GO相似性图的DataFrame转换为图结构
    """

    def __init__(self, edge_list, gene_list, node_map):
        """
        参数:
            edge_list: DataFrame, 包含 [source, target, importance] 三列
            gene_list: list, 基因名称列表
            node_map: dict, 基因名到索引的映射 {gene_name: index}
        """
        self.edge_list = edge_list
        self.node_map = node_map

        # 构建边索引和权重
        edge_index_list = []
        edge_weight_list = []

        for _, row in edge_list.iterrows():
            source = row['source']
            target = row['target']
            weight = row['importance']

            if source in node_map and target in node_map:
                source_idx = node_map[source]
                target_idx = node_map[target]

                edge_index_list.append([source_idx, target_idx])
                edge_weight_list.append(weight)

        self.edge_index = edge_index_list
        self.edge_weight = edge_weight_list
        self.num_nodes = len(gene_list)


def build_go_adjacency_matrix(edge_list, gene_list, node_map):
    """
    将GO图转换为归一化邻接矩阵（Keras/TF1.15版本）
    等价于GEARS中SGConv的预处理
    """
    from scipy.sparse import csr_matrix, eye, diags
    import numpy as np

    num_genes = len(gene_list)

    # 1. 构建邻接矩阵
    row_indices = []
    col_indices = []
    weights = []

    for _, row in edge_list.iterrows():
        source = row['source']
        target = row['target']
        weight = row['importance']

        if source in node_map and target in node_map:
            source_idx = node_map[source]
            target_idx = node_map[target]

            row_indices.append(source_idx)
            col_indices.append(target_idx)
            weights.append(weight)

    adj_matrix = csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(num_genes, num_genes)
    )

    # 2. 对称化（取平均）
    adj_matrix = (adj_matrix + adj_matrix.T) / 2

    # 3. 添加自环（等价于SGConv的 Â = A + I）
    adj_matrix = adj_matrix + eye(num_genes, format='csr')

    # 4. 对称归一化 D^(-1/2) * Â * D^(-1/2)
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    d_inv_sqrt = diags(1.0 / np.sqrt(degrees), format='csr')
    adj_matrix = d_inv_sqrt @ adj_matrix @ d_inv_sqrt

    return adj_matrix


def parse_single_pert(i):
    """解析单基因扰动，如 'KLF1+ctrl' -> 'KLF1'"""
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    """解析双基因扰动，如 'KLF1+MAP2K6' -> ('KLF1', 'MAP2K6')"""
    return i.split('+')[0], i.split('+')[1]


def parse_any_pert(p):
    """统一解析接口，返回基因列表"""
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]


class PerturbationDataset:
    """
    训练数据集类

    负责：
    1. 分离ctrl和扰动数据
    2. 按扰动条件组织数据
    3. 构建用于PaWine的智能批次
    4. 提供训练数据生成器
    """

    def __init__(self, adata, condition_key='condition', ctrl_key='control',
                 pert_key='perturbation', nperts_key='nperts'):
        """
        参数:
            adata: AnnData对象
            condition_key: str, 完整条件名的列（如'KLF1+ctrl'）
            ctrl_key: str, 标识ctrl细胞的列（值为1表示ctrl，0表示扰动）
            pert_key: str, 扰动类型的列（如'KLF1', 'KLF1+MAP2K6', 'control'）
            nperts_key: str, 扰动基因数量的列（0=ctrl, 1=单基因, 2=双基因）
        """
        from scipy import sparse

        self.adata = adata
        self.condition_key = condition_key
        self.ctrl_key = ctrl_key
        self.pert_key = pert_key
        self.nperts_key = nperts_key
        self.is_sparse = sparse.issparse(adata.X)

        # 分离ctrl和扰动数据
        self._separate_ctrl_and_pert()

    def _separate_ctrl_and_pert(self):
        """分离ctrl细胞和扰动细胞"""

        # ctrl细胞
        ctrl_mask = self.adata.obs[self.ctrl_key] == 1
        self.ctrl_indices = np.where(ctrl_mask)[0]
        self.ctrl_expr = (self.adata.X[self.ctrl_indices].A
                          if self.is_sparse else self.adata.X[self.ctrl_indices])
        self.n_ctrl = len(self.ctrl_indices)

        # 扰动细胞，按条件组织
        pert_mask = self.adata.obs[self.ctrl_key] == 0
        self.pert_data = {}  # {condition_name: {'indices': [], 'expr': ndarray, 'nperts': int, 'genes': str}}

        for idx in np.where(pert_mask)[0]:
            # 使用condition_key作为条件名（如'KLF1+ctrl'）
            cond_name = self.adata.obs[self.condition_key].iloc[idx]
            pert_name = self.adata.obs[self.pert_key].iloc[idx]
            nperts = self.adata.obs[self.nperts_key].iloc[idx]

            if pert_name == 'control':
                continue

            if cond_name not in self.pert_data:
                self.pert_data[cond_name] = {
                    'indices': [],
                    'pert_name': pert_name,  # 实际扰动基因（如'KLF1'或'KLF1+MAP2K6'）
                    'nperts': nperts
                }
            self.pert_data[cond_name]['indices'].append(idx)

        # 提取每个条件的表达矩阵
        for cond_name in self.pert_data:
            indices = self.pert_data[cond_name]['indices']
            expr = (self.adata.X[indices].A
                    if self.is_sparse else self.adata.X[indices])
            self.pert_data[cond_name]['expr'] = expr

        # 所有扰动条件列表
        self.conditions = list(self.pert_data.keys())
        self.n_conditions = len(self.conditions)

        # 按nperts分类条件（用于智能批次）
        self.single_gene_conditions = [c for c in self.conditions
                                       if self.pert_data[c]['nperts'] == 1]
        self.multi_gene_conditions = [c for c in self.conditions
                                      if self.pert_data[c]['nperts'] >= 2]

        print(f"数据集统计:")
        print(f"  ctrl细胞数: {self.n_ctrl}")
        print(f"  扰动条件数: {self.n_conditions}")
        print(f"  单基因扰动条件: {len(self.single_gene_conditions)}")
        print(f"  多基因扰动条件: {len(self.multi_gene_conditions)}")

    def build_smart_batches_for_pawine(self, go_similarity_dict, batch_size=512,
                                       single_gene_ratio=0.6, multi_gene_ratio=0.3,
                                       ctrl_ratio=0.1, similar_gene_ratio=0.65,
                                       go_similarity_threshold=0.15):
        """
        构建用于PaWine的智能批次

        与原来的区别：
        - 原来：每个批次采样扰动后细胞
        - 现在：每个批次定义要训练的扰动条件组合，训练时再采样ctrl和对应扰动细胞

        返回:
            list of dicts，每个dict包含:
            {
                'conditions': [条件名列表],  # 该批次要训练的扰动条件
                'single_conditions': [单基因条件],
                'multi_conditions': [多基因条件],
                'pawine_pairs': [(anchor_cond, pos_cond, neg_cond, weight), ...]  # 预定义的三元组
            }
        """
        self.go_similarity_dict = go_similarity_dict
        smart_batches = []

        # 计算每批次各类型的数量
        n_single = int(batch_size * single_gene_ratio)
        n_multi = int(batch_size * multi_gene_ratio)
        n_ctrl_samples = batch_size - n_single - n_multi  # ctrl样本数（实际是为每个扰动配对）

        # 单基因条件中，GO相似的比例
        n_similar = int(n_single * similar_gene_ratio)
        n_dissimilar = n_single - n_similar

        # 按GO相似性分组单基因条件
        go_groups = self._group_conditions_by_go(go_similarity_dict, go_similarity_threshold)

        # 构建批次
        # 简化版：每个批次随机选择条件组合
        n_batches = max(1, self.n_conditions // 10)  # 大约每10个条件一个批次

        for batch_idx in range(n_batches):
            batch_info = {
                'single_conditions': [],
                'multi_conditions': [],
                'pawine_triplets': []  # (anchor_cond, pos_cond, neg_cond, weight)
            }

            # 选择单基因条件
            if self.single_gene_conditions:
                # 尝试选择GO相似的条件组
                selected_singles = self._select_similar_conditions(
                    go_groups, n_similar, n_dissimilar
                )
                batch_info['single_conditions'] = selected_singles

                # 为单基因条件构建PaWine三元组
                pawine_triplets = self._build_pawine_triplets_for_conditions(
                    selected_singles, go_similarity_dict
                )
                batch_info['pawine_triplets'] = pawine_triplets

            # 选择多基因条件
            if self.multi_gene_conditions and n_multi > 0:
                n_multi_select = min(n_multi, len(self.multi_gene_conditions))
                batch_info['multi_conditions'] = list(np.random.choice(
                    self.multi_gene_conditions, n_multi_select, replace=False
                ))

            batch_info['conditions'] = (batch_info['single_conditions'] +
                                        batch_info['multi_conditions'])

            if batch_info['conditions']:
                smart_batches.append(batch_info)

        self.smart_batches = smart_batches
        print(f"智能批次构建完成: {len(smart_batches)} 个批次")

        return smart_batches

    def _group_conditions_by_go(self, go_similarity_dict, threshold):
        """按GO相似性对单基因条件分组"""
        groups = {}
        visited = set()

        for cond in self.single_gene_conditions:
            if cond in visited:
                continue

            # 获取该条件的扰动基因
            pert_name = self.pert_data[cond]['pert_name']

            if pert_name in go_similarity_dict:
                # 找到GO相似的基因
                similar_genes = [g for g, w in go_similarity_dict[pert_name] if w >= threshold]

                # 找到对应的条件
                group = [cond]
                for other_cond in self.single_gene_conditions:
                    if other_cond != cond and other_cond not in visited:
                        other_pert = self.pert_data[other_cond]['pert_name']
                        if other_pert in similar_genes:
                            group.append(other_cond)
                            visited.add(other_cond)

                if len(group) > 1:
                    groups[pert_name] = group
                    visited.add(cond)

        return groups

    def _select_similar_conditions(self, go_groups, n_similar, n_dissimilar):
        """选择GO相似和不相似的条件"""
        selected = []

        # 选择相似条件（从同一GO组）
        if go_groups:
            group_name = np.random.choice(list(go_groups.keys()))
            group_conds = go_groups[group_name]
            n_from_group = min(n_similar, len(group_conds))
            selected.extend(np.random.choice(group_conds, n_from_group, replace=False))

        # 选择不相似条件（从其他条件中随机选）
        remaining = [c for c in self.single_gene_conditions if c not in selected]
        n_remaining = min(n_dissimilar, len(remaining))
        if n_remaining > 0:
            selected.extend(np.random.choice(remaining, n_remaining, replace=False))

        return list(selected)

    def _build_pawine_triplets_for_conditions(self, conditions, go_similarity_dict):
        """为条件列表构建PaWine三元组"""
        triplets = []

        for anchor_cond in conditions:
            anchor_pert = self.pert_data[anchor_cond]['pert_name']

            if anchor_pert not in go_similarity_dict:
                continue

            similar_genes = go_similarity_dict[anchor_pert]
            if not similar_genes:
                continue

            # 找正样本：GO相似的条件
            for pos_gene, weight in similar_genes:
                # 找到以pos_gene为扰动的条件
                pos_cond = None
                for c in conditions:
                    if c != anchor_cond and self.pert_data[c]['pert_name'] == pos_gene:
                        pos_cond = c
                        break

                if pos_cond is None:
                    continue

                # 找负样本：GO不相似的条件
                similar_gene_names = {g for g, _ in similar_genes}
                neg_candidates = [c for c in conditions
                                  if c != anchor_cond and c != pos_cond
                                  and self.pert_data[c]['pert_name'] not in similar_gene_names]

                if neg_candidates:
                    neg_cond = np.random.choice(neg_candidates)
                    triplets.append((anchor_cond, pos_cond, neg_cond, weight))
                    break  # 每个anchor只添加一个三元组

        return triplets

    def create_train_generator(self, condition_encoder, batch_size=512,
                               use_smart_batches=True):
        """
        创建训练数据生成器

        参数:
            condition_encoder: dict, 条件名到索引的映射
            batch_size: int, 批次大小
            use_smart_batches: bool, 是否使用智能批次

        生成:
            x_batch: [ctrl_expr, encoder_labels, decoder_labels, pert_expr_true]
            y_batch: [pert_expr_true, pawine_input]
        """

        def generator():
            while True:
                if use_smart_batches and hasattr(self, 'smart_batches') and self.smart_batches:
                    # 使用智能批次
                    for batch_info in np.random.permutation(self.smart_batches):
                        yield from self._generate_from_batch_info(
                            batch_info, condition_encoder, batch_size
                        )
                else:
                    # 按扰动条件轮询
                    shuffled_conditions = np.random.permutation(self.conditions)
                    for cond_name in shuffled_conditions:
                        yield self._generate_single_condition_batch(
                            cond_name, condition_encoder, batch_size
                        )

        return generator

    def _generate_from_batch_info(self, batch_info, condition_encoder, batch_size):
        """从智能批次信息生成训练数据"""
        conditions = batch_info['conditions']
        # pawine_triplets = batch_info.get('pawine_triplets', [])

        if not conditions:
            return

        # 计算每个条件采样多少细胞
        n_per_condition = max(1, batch_size // len(conditions))

        all_ctrl = []
        all_pert = []
        all_indices = []
        # condition_to_batch_indices = {}  # 条件名 -> 批次内索引范围

        current_idx = 0
        for cond_name in conditions:
            pert_info = self.pert_data[cond_name]
            n_cells = len(pert_info['indices'])
            n_sample = min(n_per_condition, n_cells)

            # 采样ctrl细胞
            ctrl_sample_idx = np.random.choice(self.n_ctrl, n_sample, replace=True)
            ctrl_batch = self.ctrl_expr[ctrl_sample_idx]

            # 采样扰动细胞
            pert_sample_idx = np.random.choice(n_cells, n_sample, replace=False)
            pert_batch = pert_info['expr'][pert_sample_idx]

            # 条件索引
            cond_idx = condition_encoder.get(cond_name, 0)
            indices_batch = np.full((n_sample, 1), cond_idx, dtype=np.int32)

            all_ctrl.append(ctrl_batch)
            all_pert.append(pert_batch)
            all_indices.append(indices_batch)

            # condition_to_batch_indices[cond_name] = (current_idx, current_idx + n_sample)
            current_idx += n_sample

        # 合并
        ctrl_expr = np.vstack(all_ctrl)
        pert_expr = np.vstack(all_pert)
        pert_indices = np.vstack(all_indices)

        # batch_size_actual = len(ctrl_expr)
        #
        # # 构建PaWine输入
        # pawine_input = np.full((batch_size_actual, 4), -1.0, dtype=np.float32)
        #
        # for anchor_cond, pos_cond, neg_cond, weight in pawine_triplets:
        #     if anchor_cond in condition_to_batch_indices and \
        #             pos_cond in condition_to_batch_indices and \
        #             neg_cond in condition_to_batch_indices:
        #
        #         anchor_start, anchor_end = condition_to_batch_indices[anchor_cond]
        #         pos_start, pos_end = condition_to_batch_indices[pos_cond]
        #         neg_start, neg_end = condition_to_batch_indices[neg_cond]
        #
        #         # 为该条件的所有样本设置三元组
        #         for a_idx in range(anchor_start, anchor_end):
        #             p_idx = np.random.randint(pos_start, pos_end)
        #             n_idx = np.random.randint(neg_start, neg_end)
        #             pawine_input[a_idx] = [a_idx, p_idx, n_idx, weight]

        x_batch = [ctrl_expr, pert_indices, pert_indices, pert_expr]
        y_batch = [pert_expr, pert_expr]

        yield (x_batch, y_batch)

    def _generate_single_condition_batch(self, cond_name, condition_encoder, batch_size):
        """为单个扰动条件生成训练批次"""
        pert_info = self.pert_data[cond_name]
        n_cells = len(pert_info['indices'])
        n_sample = min(batch_size, n_cells)

        # 采样ctrl细胞
        ctrl_sample_idx = np.random.choice(self.n_ctrl, n_sample, replace=True)
        ctrl_batch = self.ctrl_expr[ctrl_sample_idx]

        # 采样扰动细胞
        pert_sample_idx = np.random.choice(n_cells, n_sample, replace=False)
        pert_batch = pert_info['expr'][pert_sample_idx]

        # 条件索引
        cond_idx = condition_encoder.get(cond_name, 0)
        pert_indices = np.full((n_sample, 1), cond_idx, dtype=np.int32)

        # PaWine输入（单条件批次不使用PaWine）
        # pawine_input = np.full((n_sample, 4), -1.0, dtype=np.float32)

        x_batch = [ctrl_batch, pert_indices, pert_indices, pert_batch]
        y_batch = [pert_batch, pert_batch]

        return (x_batch, y_batch)

    def get_validation_data(self, valid_adata, condition_encoder):
        """
        准备验证集数据

        对于验证集，我们使用ctrl平均值作为输入
        """
        from scipy import sparse
        is_sparse = sparse.issparse(valid_adata.X)

        # 分离验证集的ctrl和扰动
        ctrl_mask = valid_adata.obs[self.ctrl_key] == 1
        pert_mask = valid_adata.obs[self.ctrl_key] == 0

        if ctrl_mask.sum() > 0:
            valid_ctrl_expr = (valid_adata.X[ctrl_mask].A
                               if is_sparse else valid_adata.X[ctrl_mask])
            ctrl_mean = valid_ctrl_expr.mean(axis=0)
        else:
            # 如果验证集没有ctrl，使用训练集的ctrl均值
            ctrl_mean = self.ctrl_expr.mean(axis=0)

        # 验证集的扰动细胞
        valid_pert_expr = (valid_adata.X[pert_mask].A
                           if is_sparse else valid_adata.X[pert_mask])
        n_valid = valid_pert_expr.shape[0]

        # ctrl输入：重复ctrl均值
        ctrl_repeated = np.tile(ctrl_mean, (n_valid, 1))

        # 条件索引
        valid_conditions = valid_adata.obs[self.condition_key].iloc[np.where(pert_mask)[0]]
        valid_indices = np.array([condition_encoder.get(c, 0) for c in valid_conditions])
        valid_indices = valid_indices.reshape(-1, 1).astype(np.int32)

        x_valid = [ctrl_repeated, valid_indices, valid_indices, valid_pert_expr]
        y_valid = [valid_pert_expr, valid_pert_expr]

        return x_valid, y_valid