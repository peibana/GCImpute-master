import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="scRNA-seq data imputation using CoImpute.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sc_data', type=str,
                        default="",
                        help='(str) Folder that stores all your datasets. '
                             'For example, your expression matrix is in ./Data/sim/truecounts.csv')

    parser.add_argument('--to_dropout',
                        action='store_true', default=False,
                        # action='store_false', default=True,
                        help='(boolean, default True) If True, apply a random mask to the real single-cell datasets. ')
    parser.add_argument('--dropout_prob', type=float,
                        default=None
                        # default=0.1
                        )

    parser.add_argument('--csv_to_load',
                        # action='store_true', default=False,
                        action='store_false', default=True,
                        help='Whether input is raw count data in TSV/CSV format')
    parser.add_argument('--x10_to_load',
                        action='store_true', default=False,
                        # action='store_false', default=True,
                        help='Whether input is raw count data in 10X format')
    parser.add_argument('--txt_to_load',
                        action='store_true', default=False,
                        # action='store_false', default=True,
                        help='Whether input is raw count data in TXT format')
    parser.add_argument(
        "--subset",
        type=float,
        default=1,
        # default=0.8,
        help="Cell subset to speed up training."
             "Either a ratio (0<x<1) or a cell number (int). Default: 1 (all)")
    parser.add_argument('-t', '--transpose', dest='transpose',
                        action='store_true', default=False,
                        # action='store_false', default=True, # 行是基因 列是细胞-需要转置
                        help='Transpose input matrix (default: False)')
    parser.add_argument('--check_counts',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--filter_min_counts',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--min_gene_counts', type=int,
                        default=100,
                        help='(int, default 1) min_gene_counts')
    parser.add_argument('--min_cell_counts', type=int,
                        default=1,
                        help='(int, default 1) min_cell_counts')
    parser.add_argument('--logtrans_input',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--extract_highly_variable_genes',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--normalize_amount',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--normalize_expression_profile',
                        action='store_true', default=False,
                        # action='store_false', default=True
                        )
    parser.add_argument('--test_split',
                        action='store_true', default=False,
                        # action='store_false', default=True,
                        help='Divide the training and test sets (default: False)')

    parser.add_argument('--graph_AE_epoch', type=int,
                        default=500,
                        help='(int, default 10) Total EM epochs')
    parser.add_argument('--graph_AE_patience', type=int,
                        default=10,
                        help='(int, default 10) Specify how many epochs stop training after validation metrics no longer improve.'
                             ' If the indicator does not improve within the specified epoch number, the learning rate will be adjusted.')
    parser.add_argument('--graph_AE_learning_rate', type=float, default=1e-3,
                        help='(float, default 1e-2) Learning rate')
    parser.add_argument('--graph_AE_factor', type=float, default=1e-1,
                        help='(float, default 1e-1) The learning rate scaling factor, which is used to resize the learning rate. '
                             'When the validation metric stops improving, the learning rate is multiplied by this factor')

    parser.add_argument('--graph_AE_use_GAT',
                        # action='store_false', default=True,
                        action='store_true', default=False,
                        help='(boolean, default False) If true, will use GAT for Graph Gene layers; otherwise will use GCN layers')
    parser.add_argument('--graph_AE_ever_saving', type=int, default=100)

    # Graph Gene related
    parser.add_argument('--graph_Gene_dropout', type=float, default=0,
                        help='(int, default 0) The dropout probability for GCN or GAT')
    parser.add_argument('--graph_Gene_hid_embed1', type=int,
                        default=64,
                        help='(int, default 64) The dim for hid_embed')
    parser.add_argument('--graph_Gene_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Graph Gene embedding size')
    parser.add_argument('--graph_Gene_gat_multi_heads', type=int, default=2,
                        help='(int, default 2)')
    parser.add_argument('--graph_Gene_gat_hid_embed1', type=int,
                        default=62,
                        help='(int, default 62) The dim for hid_embed')
    parser.add_argument('--graph_Gene_gat_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Graph Gene embedding size')
    parser.add_argument('--graph_Gene_gcn_hid_embed1', type=int,
                        default=32,
                        help='(int, default 32) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Gene_gcn_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')

    # Feature AE related
    parser.add_argument('--feature_AE_dropout', type=float, default=0.2,
                        help='(int, default 0) The dropout probability.')
    parser.add_argument('--ae_multi_heads_gene', type=int, default=8,
                        help='(int, default 2)')
    parser.add_argument('--ae_multi_heads_cell', type=int, default=8,
                        help='(int, default 2)')
    parser.add_argument('--feature_AE_hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
    parser.add_argument('--feature_AE_hidden2', type=int, default=128, help='Number of units in hidden layer 2.')

    # Graph Cell related
    parser.add_argument('--graph_Cell_hid_embed1', type=int,
                        default=64,
                        help='(int, default 32) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Cell_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')
    parser.add_argument('--graph_Cell_dropout', type=float, default=0,
                        help='(int, default 0) The dropout probability for GCN or GAT')
    parser.add_argument('--graph_Cell_gcn_hid_embed1', type=int,
                        default=32,
                        help='(int, default 32) Number of units in hidden layer 1.')
    parser.add_argument('--graph_Cell_gcn_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Number of units in hidden layer 2.')
    parser.add_argument('--graph_Cell_gat_hid_embed1', type=int,
                        default=64,
                        help='(int, default 64) The dim for hid_embed')
    parser.add_argument('--graph_Cell_gat_hid_embed2', type=int,
                        default=16,
                        help='(int, default 16) Graph Gene embedding size')
    parser.add_argument('--graph_Cell_gat_multi_heads', type=int, default=2,
                        help='(int, default 2)')


    # Output related
    parser.add_argument('--output_dir', type=str,
                        default='outputs/',
                        help="(str, default 'outputs/') Folder for storing all the outputs")
    parser.add_argument('--output_raw_csv',
                        action='store_true', default=False,
                        # action='store_false', default=True,
                        help='(boolean, default False) For csv data that is not row is cell column is gene, such as 10X data, '
                             'we may need to convert it into csv data that is row is cell column is gene.')

    args = parser.parse_args()

    return args
