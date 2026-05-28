import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'], feature_data['xtest'],
        feature_data['ytrainreg'], feature_data['ytestreg'],
        feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        N = train_features.shape[0]
        val_size = int(0.2 * N)  # 20% validation -> 320 samples
        perm = np.random.permutation(N)

        val_idx   = perm[:val_size]
        train_idx = perm[val_size:]

        # Override test sets with validation sets
        test_features       = train_features[val_idx]
        test_labels_classif = train_labels_classif[val_idx]
        test_labels_reg     = train_labels_reg[val_idx]

        train_features       = train_features[train_idx]
        train_labels_classif = train_labels_classif[train_idx]
        train_labels_reg     = train_labels_reg[train_idx]

    # Normalize (fit on train only, apply to both)
    means = train_features.mean(axis=0)
    stds  = train_features.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero for constant features

    train_features = normalize_fn(train_features, means, stds)
    test_features  = normalize_fn(test_features,  means, stds)

    ## 3. Initialize the method you want to use.

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        ### WRITE YOUR CODE HERE (MLP person)
        pass

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        preds_train = method_obj.fit(train_features, train_labels_classif)
        preds_test  = method_obj.predict(test_features)

        acc_train = accuracy_fn(preds_train, train_labels_classif)
        acc_test  = accuracy_fn(preds_test,  test_labels_classif)
        f1_train  = macrof1_fn(preds_train,  train_labels_classif)
        f1_test   = macrof1_fn(preds_test,   test_labels_classif)

        print(f"\n[{args.method.upper()} | Classification]")
        print(f"  Train — Accuracy: {acc_train:.2f}%  |  Macro F1: {f1_train:.4f}")
        print(f"  {'Test ' if args.test else 'Val  '} — Accuracy: {acc_test:.2f}%  |  Macro F1: {f1_test:.4f}")

    elif args.task == "regression":
        assert args.method != "kmeans", "You should use kmeans as a classification method"

        ### WRITE YOUR CODE HERE (MLP person)

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)