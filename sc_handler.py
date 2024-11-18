import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import expon
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



def sc_load_process(data, csv_to_load=False, x10_to_load=False, txt_to_load=False, cell_subset=1, transpose=True, check_counts=True, filter_min_counts=False, logtrans_input=True, extract_highly_variable_genes=False, normalize_amount=False, normalize_expression_profile=False, test_split=True, min_gene_counts=1, min_cell_counts=1):
    logging.info('--------> Loading and Preprocessing data ...')

    if csv_to_load:
        counts = pd.read_csv(data, index_col=0)

        if cell_subset != 1:
            if cell_subset < 1:
                counts = counts.sample(frac=cell_subset)
            else:
                counts = counts.sample(cell_subset)
        if transpose:
            # 一般对于adata.X，行对应观测（即，细胞），列对应特征（即，基因）
            # counts 行是基因，列是细胞
            adata = sc.AnnData(counts.values.T)
            adata.obs_names = counts.columns
            adata.var_names = counts.index
        else:
            # counts 行是细胞，列是基因
            adata = sc.AnnData(counts.values)  # counts是DataFrame->adata是AnnData
            adata.obs_names = counts.index  # obs是细胞-index是行名
            adata.var_names = counts.columns  # var是基因-columns是列名

    elif x10_to_load:
        adata = sc.read_10x_mtx(data, var_names='gene_ids')
        adata.X = adata.X.toarray().copy()
    elif txt_to_load:
        with open(data) as f:
            row_names = [line.split('\t')[0] for line in f.readlines()[1:]]

        # 读取列名
        with open(data) as f:
            col_names = f.readline().strip().split('\t')[1:]

        # 读取数据，跳过第一行和第一列
        with open(data) as f:
            ncols = len(f.readline().split('\t'))
        counts = np.loadtxt(open(data, "rb"), delimiter="\t", skiprows=1, usecols=range(1, ncols))


        if transpose:
            # 一般对于adata.X，行对应观测（即，细胞），列对应特征（即，基因）
            # counts 行是基因，列是细胞
            adata = sc.AnnData(counts.T)
            adata.obs_names = pd.Index(col_names)
            adata.var_names = pd.Index(row_names)
        else:
            # counts 行是细胞，列是基因
            adata = sc.AnnData(counts)
            adata.obs_names = pd.Index(row_names)  # obs是细胞
            adata.var_names = pd.Index(col_names)  # var是基因

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if check_counts:
        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sp.sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

        max_value = np.max(adata.X)
        if max_value < 10:
            print("ERROR: max value = {}. Is your data log-transformed? Please provide raw counts".format(max_value))
            exit(1)

        if sum(adata.obs_names.duplicated()):
            print("ERROR: duplicated cell labels. Please provide unique cell labels.")
            exit(1)

        if sum(adata.var_names.duplicated()):
            print("ERROR: duplicated gene labels. Please provide unique gene labels.")
            exit(1)

    adata.raw = adata.copy()

    filtered_genes, filtered_cells = None, None
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=min_gene_counts)

        sc.pp.filter_cells(adata, min_counts=min_cell_counts)

        filtered_genes = set(adata.raw.var_names) - set(adata.var_names)
        filtered_genes = pd.Index(list(filtered_genes))
        filtered_cells = set(adata.raw.obs_names) - set(adata.obs_names)
        filtered_cells = pd.Index(list(filtered_cells))

    if normalize_amount:
        sc.pp.normalize_total(adata)

    if logtrans_input:
        sc.pp.log1p(adata)

    if extract_highly_variable_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            series_gene = adata.var.highly_variable
            filtered_highly_variable_genes = series_gene.index[series_gene == False]
            filtered_genes = filtered_genes.union(filtered_highly_variable_genes)
            adata = adata[:, adata.var.highly_variable]

    if normalize_expression_profile:
        sc.pp.scale(adata)


    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.5, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'
    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')

    adata.process = adata.copy()

    if filter_min_counts:
        logging.info('--------> Successfully loaded {}:{} genes and {} cells, '
                     'preprocessed {} genes and {} cells to process.'
                     '(Keep only the genes that have counts in at least {} cells '
                     'and the cells that have a certain level of expression in at least {} gene.)'
                     .format(data, adata.raw.n_vars, adata.raw.n_obs, adata.n_vars, adata.n_obs, min_gene_counts, min_cell_counts))
    else:
        logging.info('--------> Successfully loaded {}:{} genes and {} cells'.format(data, adata.n_vars, adata.n_obs))

    return adata, (filtered_genes, filtered_cells)



def dropout(X_sc, seed, dropout):
    logging.info('Applying a random mask to the real single-cell datasets ...')

    np.random.seed(seed)
    binMask = np.ones(X_sc.shape).astype(bool)
    idx = []
    for c in range(X_sc.shape[0]):
        cells_c = X_sc.X[c, :]
        ind_pos = np.arange(X_sc.shape[1])[cells_c > 0]
        cells_c_pos = cells_c[ind_pos]

        if cells_c_pos.size > 5:
            probs = expon.pdf(cells_c_pos)
            n_masked = 1 + int(dropout * len(cells_c_pos))
            if n_masked >= cells_c_pos.size:
                print("Warning: too many cells masked for gene {} ({}/{})".format(c, n_masked, cells_c_pos.size))
                n_masked = 1 + int(0.5 * cells_c_pos.size)

            masked_idx = np.random.choice(cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False)
            binMask[c, ind_pos[sorted(masked_idx)]] = False
            idx.append(ind_pos[sorted(masked_idx)])

    dropout_info = [(i, j) for i, sub_lst in enumerate(idx) for j in sub_lst]
    X_dropout = X_sc.copy()
    for i, j in dropout_info:
        X_dropout.X[i][j] = 0

    return X_dropout, dropout_info


