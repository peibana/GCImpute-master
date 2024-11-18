import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from time import time
import torch
import numpy as np
import pandas as pd
import os

from myparse import parse_args
import sc_handler
from Graph_AE import Graph_AE_handler




def RUN_MAIN():
    param = dict()
    param['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--------> Using device: {param['device']}")
    tok_start = time()

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None

    logging.info(f"graph_AE_epoch: {args.graph_AE_epoch}")
    X_sc, _ = sc_handler.sc_load_process(args.sc_data,
                                         csv_to_load=args.csv_to_load,
                                         x10_to_load=args.x10_to_load,
                                         txt_to_load=args.txt_to_load,
                                         cell_subset=args.subset,
                                         transpose=args.transpose,
                                         check_counts=args.check_counts,
                                         filter_min_counts=args.filter_min_counts,
                                         logtrans_input=args.logtrans_input,
                                         extract_highly_variable_genes=args.extract_highly_variable_genes,
                                         normalize_amount=args.normalize_amount,
                                         normalize_expression_profile=args.normalize_expression_profile,
                                         test_split=args.test_split,
                                         min_gene_counts=args.min_gene_counts,
                                         min_cell_counts=args.min_cell_counts)

    X_process = X_sc.copy()

    X_imputed = Graph_AE_handler(X_process, args, param)

    logging.info('--------> Exporting imputed expression matrix ...')
    df_imputed = pd.DataFrame(data=X_imputed, index=X_sc.obs_names, columns=X_sc.var_names)
    df_imputed.to_csv(os.path.join(args.output_dir, 'imputed.csv'))

    tok_end = time()
    time_used = tok_end - tok_start
    logging.info(f'--------> Program Finished! Total running time (seconds) = {time_used} \n')



if __name__ == '__main__':
    RUN_MAIN()






