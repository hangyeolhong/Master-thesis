import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PU-Mixup")

    # ===== dataset ===== #
    parser.add_argument("--dataset", type=str, default="Citeseer",
                        help="Choose a dataset:[Cora,Citeseer,PubMed,Chameleon,Cornell,Texas,Wisconsin]")

    # ===== model ===== #
    parser.add_argument('--l', type=int, default=2, help='# of layers')
    parser.add_argument('--model', type=str, default="GCN",
                        help="Choose a model:[MLP, GCN, GAT]")
    
    # ===== train ===== #
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--p', type=float, default=0.9, help='p%: positive, (1-p)%: transformed into unlabeled.')

    # ===== mixup ===== #
    parser.add_argument('--mixup', action='store_true', help='Whether to use Mixup')
    parser.add_argument('--m', type=int, default=2, help='[mode] 0: PP, 1: PU, 2: UU')
    parser.add_argument('--n', type=int, default=100, help='the number of mixup nodes')

    # ===== save model ===== #
    parser.add_argument('--save', default=False, help='Whether to save model')

    # ===== plot tSNE embedding ===== #
    parser.add_argument('--plt', default=False, help='Whether to plot tSNE')
    parser.add_argument('--desc', default='')

    # ===== plot graph ===== #

    return parser.parse_args()
