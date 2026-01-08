import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

from GOCVAE.utils import (remove_sparsity)

def __entropy_from_indices(indices):
    _, counts = np.unique(indices, return_counts=True)
    return entropy(counts.astype(np.int32))

def entropy_batch_mixing(adata, label_key='batch',
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100, subsample_frac=1.0):
    adata = remove_sparsity(adata)

    n_samples = adata.shape[0]
    keep_idx = np.random.choice(np.arange(n_samples), size=min(n_samples, int(subsample_frac * n_samples)),
                                replace=False)
    adata = adata[keep_idx, :]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score

def asw(adata, label_key):
    adata = remove_sparsity(adata)

    labels = adata.obs[label_key].values

    labels_encoded = LabelEncoder().fit_transform(labels)

    return silhouette_score(adata.X, labels_encoded)

def ari(adata, label_key):
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return adjusted_rand_score(labels_encoded, labels_pred)


def nmi(adata, label_key):
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return normalized_mutual_info_score(labels_encoded, labels_pred)


def non_dropout_analysis(adata, test_res):
    metric2fct = {
        'pearson': pearsonr,
        'mse': mse
    }

    pert_metric = {}

    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    if 'ensembl_id' in adata.var.columns:
        geneid2idx = dict(zip(adata.var['ensembl_id'].values, range(len(adata.var))))
    elif hasattr(adata.var.index, 'categories'):
        geneid2idx = dict(zip(adata.var.index.categories, range(len(adata.var.index.categories))))
    else:
        geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()),
                                                                                 adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]

    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]
        de_idx = [geneid2idx[i] for i in adata.uns['top_non_dropout_de_20'][pert2pert_full_id[pert]]]
        non_zero_idx = adata.uns['non_zeros_gene_idx'][pert2pert_full_id[pert]]
        non_dropout_gene_idx = adata.uns['non_dropout_gene_idx'][pert2pert_full_id[pert]]

        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(
            test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))
        frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(de_idx)
        pert_metric[pert]['frac_correct_direction_top20_non_dropout'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0]) / len(de_idx)
        pert_metric[pert]['frac_opposite_direction_top20_non_dropout'] = frac_direction_opposite

        frac_direction_opposite = len(np.where(direc_change == 1)[0]) / len(de_idx)
        pert_metric[pert]['frac_0/1_direction_top20_non_dropout'] = frac_direction_opposite

        direc_change = np.abs(
            np.sign(test_res['pred'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]) - np.sign(
                test_res['truth'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]))
        frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(non_zero_idx)
        pert_metric[pert]['frac_correct_direction_non_zero'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0]) / len(non_zero_idx)
        pert_metric[pert]['frac_opposite_direction_non_zero'] = frac_direction_opposite

        frac_direction_opposite = len(np.where(direc_change == 1)[0]) / len(non_zero_idx)
        pert_metric[pert]['frac_0/1_direction_non_zero'] = frac_direction_opposite

        direc_change = np.abs(
            np.sign(test_res['pred'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]) - np.sign(
                test_res['truth'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]))
        frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(non_dropout_gene_idx)
        pert_metric[pert]['frac_correct_direction_non_dropout'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0]) / len(non_dropout_gene_idx)
        pert_metric[pert]['frac_opposite_direction_non_dropout'] = frac_direction_opposite

        frac_direction_opposite = len(np.where(direc_change == 1)[0]) / len(non_dropout_gene_idx)
        pert_metric[pert]['frac_0/1_direction_non_dropout'] = frac_direction_opposite

        mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis=0)
        std = np.std(test_res['truth'][pert_idx][:, de_idx], axis=0)
        min_ = np.min(test_res['truth'][pert_idx][:, de_idx], axis=0)
        max_ = np.max(test_res['truth'][pert_idx][:, de_idx], axis=0)
        q25 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.25, axis=0)
        q75 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.75, axis=0)
        q55 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.55, axis=0)
        q45 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.45, axis=0)
        q40 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.4, axis=0)
        q60 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.6, axis=0)

        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)

        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:
            pred_mean = np.mean(test_res['pred'][pert_idx][:, de_idx], axis=0).reshape(-1, )
            true_mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis=0).reshape(-1, )

            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_non_dropout'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55_non_dropout'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60_non_dropout'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75_non_dropout'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx])) / (std[zero_idx])
            pert_metric[pert]['mean_sigma_non_dropout'] = np.mean(sigma)
            pert_metric[pert]['std_sigma_non_dropout'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1_non_dropout'] = 1 - len(np.where(sigma > 1)[0]) / len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2_non_dropout'] = 1 - len(np.where(sigma > 2)[0]) / len(zero_idx)

        p_idx = np.where(test_res['pert_cat'] == pert)[0]
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx],
                          test_res['truth'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de_non_dropout'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx],
                          test_res['truth'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de_non_dropout'] = val

    return pert_metric


def deeper_analysis(adata, test_res, de_column_prefix='rank_genes_groups_cov', most_variable_genes=None):
    metric2fct = {
        'pearson': pearsonr,
        'mse': mse
    }

    pert_metric = {}

    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    if 'ensembl_id' in adata.var.columns:
        geneid2idx = dict(zip(adata.var['ensembl_id'].values, range(len(adata.var))))
    else:
        geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()),
                                                                                 adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]

    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis=0))[-200:]

    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}
        de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
        de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
        de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
        de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]
        pred_mean = np.mean(test_res['pred_de'][pert_idx], axis=0).reshape(-1, )
        true_mean = np.mean(test_res['truth_de'][pert_idx], axis=0).reshape(-1, )

        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0) - ctrl[0]) - np.sign(
            test_res['truth'][pert_idx].mean(0) - ctrl[0]))
        frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(geneid2name)
        pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction

        de_idx_map = {20: de_idx,
                      50: de_idx_50,
                      100: de_idx_100,
                      200: de_idx_200
                      }

        for val in [20, 50, 100, 200]:
            direc_change = np.abs(
                np.sign(test_res['pred'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]) - np.sign(
                    test_res['truth'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]))
            frac_correct_direction = len(np.where(direc_change == 0)[0]) / val
            pert_metric[pert]['frac_correct_direction_' + str(val)] = frac_correct_direction

        mean = np.mean(test_res['truth_de'][pert_idx], axis=0)
        std = np.std(test_res['truth_de'][pert_idx], axis=0)
        min_ = np.min(test_res['truth_de'][pert_idx], axis=0)
        max_ = np.max(test_res['truth_de'][pert_idx], axis=0)
        q25 = np.quantile(test_res['truth_de'][pert_idx], 0.25, axis=0)
        q75 = np.quantile(test_res['truth_de'][pert_idx], 0.75, axis=0)
        q55 = np.quantile(test_res['truth_de'][pert_idx], 0.55, axis=0)
        q45 = np.quantile(test_res['truth_de'][pert_idx], 0.45, axis=0)
        q40 = np.quantile(test_res['truth_de'][pert_idx], 0.4, axis=0)
        q60 = np.quantile(test_res['truth_de'][pert_idx], 0.6, axis=0)

        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:

            direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) - np.sign(
                true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))
            frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(nonzero_des)
            pert_metric[pert]['frac_correct_direction_20_nonzero'] = frac_correct_direction

            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range) / len(nonzero_des)
            pert_metric[pert]['frac_in_range'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25) / len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx])) / (std[zero_idx])
            pert_metric[pert]['mean_sigma'] = np.mean(sigma)
            pert_metric[pert]['std_sigma'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1'] = 1 - len(np.where(sigma > 1)[0]) / len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2'] = 1 - len(np.where(sigma > 2)[0]) / len(zero_idx)

        ## correlation on delta
        p_idx = np.where(test_res['pert_cat'] == pert)[0]

        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0) - ctrl[0], test_res['truth'][p_idx].mean(0) - ctrl[0])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx],
                          test_res['truth'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta_de'] = val

        ## up fold changes > 10?
        pert_mean = np.mean(test_res['truth'][p_idx], axis=0).reshape(-1, )

        fold_change = pert_mean / ctrl
        fold_change[np.isnan(fold_change)] = 0
        fold_change[np.isinf(fold_change)] = 0
        ## this is to remove the ones that are super low and the fold change becomes unmeaningful
        fold_change[0][np.where(pert_mean < 0.5)[0]] = 0

        o = np.where(fold_change[0] > 0)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_all'] = np.mean(np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc))

        o = np.intersect1d(np.where(fold_change[0] < 0.333)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.33'] = np.mean(np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc))

        o = np.intersect1d(np.where(fold_change[0] < 0.1)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.1'] = np.mean(np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc))

        o = np.where(fold_change[0] > 3)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_3'] = np.mean(np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc))

        o = np.where(fold_change[0] > 10)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_10'] = np.mean(np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc))

        ## most variable genes
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes] - ctrl[0][most_variable_genes],
                          test_res['truth'][p_idx].mean(0)[most_variable_genes] - ctrl[0][most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_hvg'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes],
                          test_res['truth'][p_idx].mean(0)[most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_hvg'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes],
                          test_res['truth'][p_idx].mean(0)[most_variable_genes])
                pert_metric[pert][m + '_top200_hvg'] = val

        ## top 20/50/100/200 DEs
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx],
                          test_res['truth'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx],
                          test_res['truth'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de'] = val
                pert_metric[pert]['mse_top20_de'] = val

        # ===== Accuracy=====
        pert_full_id = pert2pert_full_id[pert]
        if 'top_non_dropout_de_20' in adata.uns and pert_full_id in adata.uns['top_non_dropout_de_20']:

            de_idx_nondropout = [geneid2idx[i] for i in adata.uns['top_non_dropout_de_20'][pert_full_id]]
            idx = de_idx_nondropout[:20]  # 取前20个（与train1.py的 idx = de_gene_idx[:20] 一致）

            pred_mean_de = test_res['pred'][p_idx].mean(0)[idx]
            truth_mean_de = test_res['truth'][p_idx].mean(0)[idx]

            pert_metric[pert]['accuracy_0.01'] = compute_percent(truth_mean_de, pred_mean_de, 0.01)
            pert_metric[pert]['accuracy_0.05'] = compute_percent(truth_mean_de, pred_mean_de, 0.05)

        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200],
                          test_res['truth'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_de'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200],
                          test_res['truth'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200])
                pert_metric[pert][m + '_top200_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100],
                          test_res['truth'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top100_de'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top100_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100],
                          test_res['truth'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100])
                pert_metric[pert][m + '_top100_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50],
                          test_res['truth'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top50_de'] = val

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top50_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50],
                          test_res['truth'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50])
                pert_metric[pert][m + '_top50_de'] = val

    return pert_metric


def GI_subgroup(pert_metric):
    GI_type2Score = {}
    test_pert_list = list(pert_metric.keys())
    for GI_type, gi_list in GIs.items():
        intersect = np.intersect1d(gi_list, test_pert_list)
        if len(intersect) != 0:
            GI_type2Score[GI_type] = {}

            for m in list(list(pert_metric.values())[0].keys()):
                GI_type2Score[GI_type][m] = np.mean([pert_metric[i][m] for i in intersect if m in pert_metric[i]])

    return GI_type2Score


GIs = {
    'NEOMORPHIC': ['CBL+TGFBR2',
                  'KLF1+TGFBR2',
                  'MAP2K6+SPI1',
                  'SAMD1+TGFBR2',
                  'TGFBR2+C19orf26',
                  'TGFBR2+ETS2',
                  'CBL+UBASH3A',
                  'CEBPE+KLF1',
                  'DUSP9+MAPK1',
                  'FOSB+PTPN12',
                  'PLK4+STIL',
                  'PTPN12+OSR2',
                  'ZC3HAV1+CEBPE'],
    'ADDITIVE': ['BPGM+SAMD1',
                'CEBPB+MAPK1',
                'CEBPB+OSR2',
                'DUSP9+PRTG',
                'FOSB+OSR2',
                'IRF1+SET',
                'MAP2K3+ELMSAN1',
                'MAP2K6+ELMSAN1',
                'POU3F2+FOXL2',
                'RHOXF2BB+SET',
                'SAMD1+PTPN12',
                'SAMD1+UBASH3B',
                'SAMD1+ZBTB1',
                'SGK1+TBX2',
                'TBX3+TBX2',
                'ZBTB10+SNAI1'],
    'EPISTASIS': ['AHR+KLF1',
                 'MAPK1+TGFBR2',
                 'TGFBR2+IGDCC3',
                 'TGFBR2+PRTG',
                 'UBASH3B+OSR2',
                 'DUSP9+ETS2',
                 'KLF1+CEBPA',
                 'MAP2K6+IKZF3',
                 'ZC3HAV1+CEBPA'],
    'REDUNDANT': ['CDKN1C+CDKN1A',
                 'MAP2K3+MAP2K6',
                 'CEBPB+CEBPA',
                 'CEBPE+CEBPA',
                 'CEBPE+SPI1',
                 'ETS2+MAPK1',
                 'FOSB+CEBPE',
                 'FOXA3+FOXA1'],
    'POTENTIATION': ['CNN1+UBASH3A',
                    'ETS2+MAP7D1',
                    'FEV+CBFA2T3',
                    'FEV+ISL2',
                    'FEV+MAP7D1',
                    'PTPN12+UBASH3A'],
    'SYNERGY_SIMILAR_PHENO':['CBL+CNN1',
                            'CBL+PTPN12',
                            'CBL+PTPN9',
                            'CBL+UBASH3B',
                            'FOXA3+FOXL2',
                            'FOXA3+HOXB9',
                            'FOXL2+HOXB9',
                            'UBASH3B+CNN1',
                            'UBASH3B+PTPN12',
                            'UBASH3B+PTPN9',
                            'UBASH3B+ZBTB25'],
    'SYNERGY_DISSIMILAR_PHENO': ['AHR+FEV',
                                'DUSP9+SNAI1',
                                'FOXA1+FOXF1',
                                'FOXA1+FOXL2',
                                'FOXA1+HOXB9',
                                'FOXF1+FOXL2',
                                'FOXF1+HOXB9',
                                'FOXL2+MEIS1',
                                'IGDCC3+ZBTB25',
                                'POU3F2+CBFA2T3',
                                'PTPN12+ZBTB25',
                                'SNAI1+DLX2',
                                'SNAI1+UBASH3B'],
    'SUPPRESSOR': ['CEBPB+PTPN12',
                  'CEBPE+CNN1',
                  'CEBPE+PTPN12',
                  'CNN1+MAPK1',
                  'ETS2+CNN1',
                  'ETS2+IGDCC3',
                  'ETS2+PRTG',
                  'FOSB+UBASH3B',
                  'IGDCC3+MAPK1',
                  'LYL1+CEBPB',
                  'MAPK1+PRTG',
                  'PTPN12+SNAI1']
}


def compute_percent(y_true, y_pred, threshold):
    """
    Calculate the proportion of predicted values within ±threshold% of the actual values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    abs_error = np.abs(y_pred - y_true)

    tolerance = threshold * np.abs(y_true)

    within_range = abs_error <= tolerance

    proportion = np.sum(within_range) / len(y_true)

    return proportion