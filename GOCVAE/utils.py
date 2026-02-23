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
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def label_encoder(adata, le=None, condition_key='condition'):
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


def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list


def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = os.path.join(data_path, f'go_essential_{data_name}.csv')
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list if i in gene2go}

    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                 total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})

    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list


def get_similarity_network(k, data_path, data_name, pert_list=None):
    """
    Construct a GO functional similarity gene-gene network diagram.
    """
    df_jaccard = make_GO(data_path, pert_list, data_name)

    df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1, ['importance'])).reset_index(drop=True)

    return df_out

def convert_ensembl_to_symbol(ensembl_ids):
    try:
        import mygene
        mg = mygene.MyGeneInfo()

        results = mg.querymany(
            ensembl_ids,
            scopes='ensembl.gene',
            fields='symbol',
            species='human'     
        )

        converted_genes = []
        for result in results:
            if 'symbol' in result and result['symbol']:
                converted_genes.append(result['symbol'])

        return converted_genes

    except ImportError:
        return []
    except Exception as e:
        return []


def filter_genes_with_go_annotation(gene_list, data_path):
    """
    Filter out genes without GO annotations
    """
    gene2go_file = os.path.join(data_path, 'gene2go_all.pkl')

    with open(gene2go_file, 'rb') as f:
        gene2go = pickle.load(f)

    available_genes = set(gene2go.keys())
    filtered_genes = [gene for gene in gene_list if gene in available_genes]

    return filtered_genes


def prepare_gene_embeddings_for_model(adata, condition_key='condition',
                                    data_path='../data/GEARS/',
                                    go_network_file=None,
                                    output_dir='../data/GEARS/test_pawine/',
                                    embedding_dim=32):

    from GOCVAE.utils import convert_ensembl_to_symbol, filter_genes_with_go_annotation

    ensembl_ids = adata.var_names.tolist()
    converted_genes = convert_ensembl_to_symbol(ensembl_ids)
    gene_list = filter_genes_with_go_annotation(converted_genes, data_path)

    if go_network_file is None:
        go_network_file = os.path.join(data_path, 'GEARS', 'GO', 'go_graph_sciplex_test_pawine.txt')

    embedding_path = run_pawine_training(
        go_network_file=go_network_file,
        gene_list=gene_list,
        embedding_dim=embedding_dim,
        output_dir=output_dir
    )

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


def ensure_required_columns(adata):
    if 'nperts' not in adata.obs.columns:

        def count_perts(condition):
            if condition == 'ctrl':
                return 0
            clean = condition.replace('+ctrl', '').replace('ctrl+', '')
            return 1 if '+' not in clean else clean.count('+') + 1

        adata.obs['nperts'] = adata.obs['condition'].apply(count_perts)

    if 'perturbation' not in adata.obs.columns:

        def extract_pert(condition):
            if condition == 'ctrl':
                return 'control'
            return condition.replace('+ctrl', '').replace('ctrl+', '')

        adata.obs['perturbation'] = adata.obs['condition'].apply(extract_pert)

    return adata



def create_smart_batches(adata, batch_size=512, data_path='../data/GEARS/', go_similarity_dict=None,
                         single_gene_ratio=0.6, multi_gene_ratio=0.3, control_ratio=0.1,
                         go_similarity_threshold=0.15, min_go_group_size=10,
                         similar_gene_ratio=0.7):
    """
    Create Smart Batch Grouping for PaWine Training
    """
    adata = ensure_required_columns(adata)

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

    if go_similarity_dict is not None:
        print(f"Using the pre-built GO network: {len(go_similarity_dict)} nodes")
    else:
        print("GO network not provided, beginning construction...")
        if 'symbol' in adata.var.columns:
            all_genes = adata.var['symbol'].unique().tolist()
        elif 'gene_name' in adata.var.columns:
            all_genes = adata.var['gene_name'].unique().tolist()
        else:
            all_genes = adata.var_names.tolist()

        filtered_genes = filter_genes_with_go_annotation(all_genes, data_path)

        if len(filtered_genes) < min_go_group_size:
            go_similarity_dict = {}
        else:
            go_df = get_similarity_network(k=20, data_path=data_path,
                                           data_name='norman2019_full', pert_list=filtered_genes)

            go_df_filtered = go_df[go_df['importance'] >= go_similarity_threshold]

            go_similarity_dict = {}
            for _, row in go_df_filtered.iterrows():
                source, target, weight = row['source'], row['target'], row['importance']
                if source not in go_similarity_dict:
                    go_similarity_dict[source] = []
                go_similarity_dict[source].append((target, weight))
                if target not in go_similarity_dict:
                    go_similarity_dict[target] = []
                go_similarity_dict[target].append((source, weight))

            for gene in go_similarity_dict:
                go_similarity_dict[gene].sort(key=lambda x: x[1], reverse=True)

    single_gene_data = adata[single_gene_indices]
    unique_perturbed_genes = single_gene_data.obs['perturbation'].unique()

    if len(go_similarity_dict) > 0:
        go_functional_groups = _cluster_genes_by_go_similarity(
            unique_perturbed_genes,
            go_similarity_dict,
            min_go_group_size
        )
    else:
        go_functional_groups = {'random': list(unique_perturbed_genes)}

    gene_to_sample_indices = defaultdict(list)
    for idx in single_gene_indices:
        gene = adata.obs['perturbation'].iloc[idx]
        gene_to_sample_indices[gene].append(idx)

    batch_indices_list = []

    n_single_per_batch = int(batch_size * single_gene_ratio)
    n_multi_per_batch = int(batch_size * multi_gene_ratio)

    n_similar_genes = int(n_single_per_batch * similar_gene_ratio)
    n_different_genes = n_single_per_batch - n_similar_genes

    total_batches_needed = max(1, len(single_gene_indices) // n_single_per_batch)

    for batch_idx in range(total_batches_needed):
        current_batch = []

        main_group_id = batch_idx % len(go_functional_groups)
        main_group_genes = list(go_functional_groups.values())[main_group_id]

        similar_samples = _sample_from_gene_group(main_group_genes, gene_to_sample_indices,
                                                  n_similar_genes)
        current_batch.extend(similar_samples)

        other_groups = [genes for i, genes in enumerate(go_functional_groups.values())
                        if i != main_group_id]
        if other_groups:
            other_genes = [gene for group in other_groups for gene in group]
            different_samples = _sample_from_gene_group(other_genes, gene_to_sample_indices,
                                                        n_different_genes)
            current_batch.extend(different_samples)

        if len(multi_gene_indices) >= n_multi_per_batch:
            multi_samples = np.random.choice(multi_gene_indices, n_multi_per_batch, replace=False)
            current_batch.extend(multi_samples)
        else:
            current_batch.extend(multi_gene_indices)

        remaining_needed = batch_size - len(current_batch)
        if len(control_indices) >= remaining_needed and remaining_needed > 0:
            control_samples = np.random.choice(control_indices, remaining_needed, replace=False)
            current_batch.extend(control_samples)

        current_batch = current_batch[:batch_size]
        if len(current_batch) >= batch_size * 0.8:
            batch_indices_list.append(np.array(current_batch))

    return batch_indices_list, go_similarity_dict


def _cluster_genes_by_go_similarity(genes, go_similarity_dict, min_group_size):
    """
    Clustering genes based on GO similarity
    """
    if len(genes) < min_group_size:
        return {'single_group': genes}

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

    try:
        distance_matrix = 1 - similarity_matrix
        n_clusters = max(2, min(8, n_genes // min_group_size))

        mds = MDS(n_components=min(10, n_genes - 1), dissimilarity='precomputed', random_state=42)
        gene_embeddings = mds.fit_transform(distance_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(gene_embeddings)

        groups = defaultdict(list)
        for gene, cluster_id in zip(genes, cluster_labels):
            groups[f'group_{cluster_id}'].append(gene)

        filtered_groups = {gid: genes_list for gid, genes_list in groups.items()
                           if len(genes_list) >= min_group_size}

        if not filtered_groups:
            return {'single_group': genes}

        return filtered_groups

    except Exception as e:
        return {'single_group': genes}


def _sample_from_gene_group(genes, gene_to_sample_indices, n_samples):
    """
    Sample a specified number of samples from the genome
    """
    available_samples = []
    for gene in genes:
        available_samples.extend(gene_to_sample_indices.get(gene, []))

    if len(available_samples) == 0:
        return []

    n_to_sample = min(n_samples, len(available_samples))
    return np.random.choice(available_samples, n_to_sample, replace=False).tolist()


class GeneSimNetwork:
    """
    Gene Similarity Network
    """

    def __init__(self, edge_list, gene_list, node_map):
        self.edge_list = edge_list
        self.node_map = node_map

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

    from scipy.sparse import csr_matrix, eye, diags
    import numpy as np

    num_genes = len(gene_list)

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

    adj_matrix = (adj_matrix + adj_matrix.T) / 2

    adj_matrix = adj_matrix + eye(num_genes, format='csr')

    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    d_inv_sqrt = diags(1.0 / np.sqrt(degrees), format='csr')
    adj_matrix = d_inv_sqrt @ adj_matrix @ d_inv_sqrt

    return adj_matrix


def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]


def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]


class PerturbationDataset:
    """
    Training Dataset Class
    """

    def __init__(self, adata, condition_key='condition', ctrl_key='control',
                 pert_key='perturbation', nperts_key='nperts'):

        from scipy import sparse

        self.adata = adata
        self.condition_key = condition_key
        self.ctrl_key = ctrl_key
        self.pert_key = pert_key
        self.nperts_key = nperts_key
        self.is_sparse = sparse.issparse(adata.X)

        self._separate_ctrl_and_pert()

    def _separate_ctrl_and_pert(self):

        ctrl_mask = self.adata.obs[self.ctrl_key] == 1
        self.ctrl_indices = np.where(ctrl_mask)[0]
        self.ctrl_expr = (self.adata.X[self.ctrl_indices].A
                          if self.is_sparse else self.adata.X[self.ctrl_indices])
        self.n_ctrl = len(self.ctrl_indices)

        pert_mask = self.adata.obs[self.ctrl_key] == 0
        self.pert_data = {}  # {condition_name: {'indices': [], 'expr': ndarray, 'nperts': int, 'genes': str}}

        for idx in np.where(pert_mask)[0]:
            cond_name = self.adata.obs[self.condition_key].iloc[idx]
            pert_name = self.adata.obs[self.pert_key].iloc[idx]
            nperts = self.adata.obs[self.nperts_key].iloc[idx]

            if pert_name == 'control':
                continue

            if cond_name not in self.pert_data:
                self.pert_data[cond_name] = {
                    'indices': [],
                    'pert_name': pert_name,
                    'nperts': nperts
                }
            self.pert_data[cond_name]['indices'].append(idx)

        for cond_name in self.pert_data:
            indices = self.pert_data[cond_name]['indices']
            expr = (self.adata.X[indices].A
                    if self.is_sparse else self.adata.X[indices])
            self.pert_data[cond_name]['expr'] = expr

        self.conditions = list(self.pert_data.keys())
        self.n_conditions = len(self.conditions)

        self.single_gene_conditions = [c for c in self.conditions
                                       if self.pert_data[c]['nperts'] == 1]
        self.multi_gene_conditions = [c for c in self.conditions
                                      if self.pert_data[c]['nperts'] >= 2]


    def build_smart_batches_for_pawine(self, go_similarity_dict, batch_size=512,
                                       single_gene_ratio=0.6, multi_gene_ratio=0.3,
                                       ctrl_ratio=0.1, similar_gene_ratio=0.65,
                                       go_similarity_threshold=0.15):
        """
        Building Smart Batches for PaWine
        """
        self.go_similarity_dict = go_similarity_dict
        smart_batches = []

        n_single = int(batch_size * single_gene_ratio)
        n_multi = int(batch_size * multi_gene_ratio)

        n_similar = int(n_single * similar_gene_ratio)
        n_dissimilar = n_single - n_similar

        go_groups = self._group_conditions_by_go(go_similarity_dict, go_similarity_threshold)

        n_batches = max(1, self.n_conditions // 10)  

        for batch_idx in range(n_batches):
            batch_info = {
                'single_conditions': [],
                'multi_conditions': [],
                'pawine_triplets': []  
            }

            if self.single_gene_conditions:
                selected_singles = self._select_similar_conditions(
                    go_groups, n_similar, n_dissimilar
                )
                batch_info['single_conditions'] = selected_singles

                pawine_triplets = self._build_pawine_triplets_for_conditions(
                    selected_singles, go_similarity_dict
                )
                batch_info['pawine_triplets'] = pawine_triplets

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

        return smart_batches

    def _group_conditions_by_go(self, go_similarity_dict, threshold):
        groups = {}
        visited = set()

        for cond in self.single_gene_conditions:
            if cond in visited:
                continue

            pert_name = self.pert_data[cond]['pert_name']

            if pert_name in go_similarity_dict:
                similar_genes = [g for g, w in go_similarity_dict[pert_name] if w >= threshold]

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
        selected = []

        if go_groups:
            group_name = np.random.choice(list(go_groups.keys()))
            group_conds = go_groups[group_name]
            n_from_group = min(n_similar, len(group_conds))
            selected.extend(np.random.choice(group_conds, n_from_group, replace=False))

        remaining = [c for c in self.single_gene_conditions if c not in selected]
        n_remaining = min(n_dissimilar, len(remaining))
        if n_remaining > 0:
            selected.extend(np.random.choice(remaining, n_remaining, replace=False))

        return list(selected)

    def _build_pawine_triplets_for_conditions(self, conditions, go_similarity_dict):
        triplets = []

        for anchor_cond in conditions:
            anchor_pert = self.pert_data[anchor_cond]['pert_name']

            if anchor_pert not in go_similarity_dict:
                continue

            similar_genes = go_similarity_dict[anchor_pert]
            if not similar_genes:
                continue

            for pos_gene, weight in similar_genes:
                pos_cond = None
                for c in conditions:
                    if c != anchor_cond and self.pert_data[c]['pert_name'] == pos_gene:
                        pos_cond = c
                        break

                if pos_cond is None:
                    continue

                similar_gene_names = {g for g, _ in similar_genes}
                neg_candidates = [c for c in conditions
                                  if c != anchor_cond and c != pos_cond
                                  and self.pert_data[c]['pert_name'] not in similar_gene_names]

                if neg_candidates:
                    neg_cond = np.random.choice(neg_candidates)
                    triplets.append((anchor_cond, pos_cond, neg_cond, weight))
                    break  

        return triplets

    def create_train_generator(self, condition_encoder, batch_size=512,
                               use_smart_batches=True):

        def generator():
            while True:
                if use_smart_batches and hasattr(self, 'smart_batches') and self.smart_batches:
                    for batch_info in np.random.permutation(self.smart_batches):
                        yield from self._generate_from_batch_info(
                            batch_info, condition_encoder, batch_size
                        )
                else:
                    shuffled_conditions = np.random.permutation(self.conditions)
                    for cond_name in shuffled_conditions:
                        yield self._generate_single_condition_batch(
                            cond_name, condition_encoder, batch_size
                        )

        return generator

    def _generate_from_batch_info(self, batch_info, condition_encoder, batch_size):
        conditions = batch_info['conditions']
        pawine_triplets = batch_info.get('pawine_triplets', [])

        if not conditions:
            return

        n_per_condition = max(1, batch_size // len(conditions))

        all_ctrl = []
        all_pert = []
        all_indices = []
        condition_to_batch_indices = {}  

        current_idx = 0
        for cond_name in conditions:
            pert_info = self.pert_data[cond_name]
            n_cells = len(pert_info['indices'])
            n_sample = min(n_per_condition, n_cells)

            ctrl_sample_idx = np.random.choice(self.n_ctrl, n_sample, replace=True)
            ctrl_batch = self.ctrl_expr[ctrl_sample_idx]

            pert_sample_idx = np.random.choice(n_cells, n_sample, replace=False)
            pert_batch = pert_info['expr'][pert_sample_idx]

            cond_idx = condition_encoder.get(cond_name, 0)
            indices_batch = np.full((n_sample, 1), cond_idx, dtype=np.int32)

            all_ctrl.append(ctrl_batch)
            all_pert.append(pert_batch)
            all_indices.append(indices_batch)

            condition_to_batch_indices[cond_name] = (current_idx, current_idx + n_sample)
            current_idx += n_sample

        ctrl_expr = np.vstack(all_ctrl)
        pert_expr = np.vstack(all_pert)
        pert_indices = np.vstack(all_indices)

        batch_size_actual = len(ctrl_expr)

        pawine_input = np.full((batch_size_actual, 4), -1.0, dtype=np.float32)

        for anchor_cond, pos_cond, neg_cond, weight in pawine_triplets:
            if anchor_cond in condition_to_batch_indices and \
                    pos_cond in condition_to_batch_indices and \
                    neg_cond in condition_to_batch_indices:

                anchor_start, anchor_end = condition_to_batch_indices[anchor_cond]
                pos_start, pos_end = condition_to_batch_indices[pos_cond]
                neg_start, neg_end = condition_to_batch_indices[neg_cond]

                for a_idx in range(anchor_start, anchor_end):
                    p_idx = np.random.randint(pos_start, pos_end)
                    n_idx = np.random.randint(neg_start, neg_end)
                    pawine_input[a_idx] = [a_idx, p_idx, n_idx, weight]

        x_batch = [ctrl_expr, pert_indices, pert_indices, pert_expr]
        y_batch = [pert_expr, pawine_input, pert_expr]

        yield (x_batch, y_batch)

    def _generate_single_condition_batch(self, cond_name, condition_encoder, batch_size):

        pert_info = self.pert_data[cond_name]
        n_cells = len(pert_info['indices'])
        n_sample = min(batch_size, n_cells)

        ctrl_sample_idx = np.random.choice(self.n_ctrl, n_sample, replace=True)
        ctrl_batch = self.ctrl_expr[ctrl_sample_idx]

        pert_sample_idx = np.random.choice(n_cells, n_sample, replace=False)
        pert_batch = pert_info['expr'][pert_sample_idx]

        cond_idx = condition_encoder.get(cond_name, 0)
        pert_indices = np.full((n_sample, 1), cond_idx, dtype=np.int32)

        pawine_input = np.full((n_sample, 4), -1.0, dtype=np.float32)

        x_batch = [ctrl_batch, pert_indices, pert_indices, pert_batch]
        y_batch = [pert_batch, pawine_input, pert_batch]

        return (x_batch, y_batch)

    def get_validation_data(self, valid_adata, condition_encoder):

        from scipy import sparse
        is_sparse = sparse.issparse(valid_adata.X)

        ctrl_mask = valid_adata.obs[self.ctrl_key] == 1
        pert_mask = valid_adata.obs[self.ctrl_key] == 0

        if ctrl_mask.sum() > 0:
            valid_ctrl_expr = (valid_adata.X[ctrl_mask].A
                               if is_sparse else valid_adata.X[ctrl_mask])
            ctrl_mean = valid_ctrl_expr.mean(axis=0)
        else:
            ctrl_mean = self.ctrl_expr.mean(axis=0)

        valid_pert_expr = (valid_adata.X[pert_mask].A
                           if is_sparse else valid_adata.X[pert_mask])
        n_valid = valid_pert_expr.shape[0]

        ctrl_repeated = np.tile(ctrl_mean, (n_valid, 1))

        valid_conditions = valid_adata.obs[self.condition_key].iloc[np.where(pert_mask)[0]]
        valid_indices = np.array([condition_encoder.get(c, 0) for c in valid_conditions])
        valid_indices = valid_indices.reshape(-1, 1).astype(np.int32)

        x_valid = [ctrl_repeated, valid_indices, valid_indices, valid_pert_expr]
        y_valid = [valid_pert_expr, np.full((n_valid, 4), -1.0, dtype=np.float32), valid_pert_expr]

        return x_valid, y_valid
