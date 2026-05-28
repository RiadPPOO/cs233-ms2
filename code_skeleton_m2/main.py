import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Softmax, Identity
from src.methods.kmeans import KMeans
from src.utils import (normalize_fn, append_bias_term, accuracy_fn,
                       macrof1_fn, mse_fn, get_n_classes,
                       label_to_onehot, onehot_to_label)
import os

np.random.seed(100)


def main(args):
    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. Load the data.
    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'], feature_data['xtest'],
        feature_data['ytrainreg'], feature_data['ytestreg'],
        feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    ## 2. Prepare it (validation split + normalization).
    if not args.test:
        N = train_features.shape[0]
        val_size = int(0.2 * N)
        perm = np.random.permutation(N)
        val_idx   = perm[:val_size]
        train_idx = perm[val_size:]

        test_features       = train_features[val_idx]
        test_labels_classif = train_labels_classif[val_idx]
        test_labels_reg     = train_labels_reg[val_idx]

        train_features       = train_features[train_idx]
        train_labels_classif = train_labels_classif[train_idx]
        train_labels_reg     = train_labels_reg[train_idx]

    means = train_features.mean(axis=0)
    stds  = train_features.std(axis=0)
    stds[stds == 0] = 1
    train_features = normalize_fn(train_features, means, stds)
    test_features  = normalize_fn(test_features,  means, stds)

    ## 3. Initialize the method.
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        D = train_features.shape[1]
        act = ReLU if args.activation == "relu" else Sigmoid
        if args.task == "classification":
            out_dim = get_n_classes(train_labels_classif)
            dimensions = [D] + args.hidden_sizes + [out_dim]
            activations = [act] * len(args.hidden_sizes) + [Softmax]
            if args.weighted:
                _, counts = np.unique(train_labels_classif, return_counts=True)
                class_weights = counts.sum() / (out_dim * counts)
            else:
                class_weights = None
            method_obj = MLP(dimensions, activations, class_weights=class_weights)
        else:
            dimensions = [D] + args.hidden_sizes + [1]
            activations = [act] * len(args.hidden_sizes) + [Identity]
            method_obj = MLP(dimensions, activations)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate.
    if args.task == "classification":
        if args.method == "mlp":
            out_dim = get_n_classes(train_labels_classif)
            y_train_oh = label_to_onehot(train_labels_classif, out_dim)
            method_obj.fit(train_features, y_train_oh, CrossEntropy,
                           epochs=args.max_iters, batch_size=args.batch_size,
                           learning_rate=args.lr)
            preds_train = onehot_to_label(method_obj.predict(train_features))
            preds_test  = onehot_to_label(method_obj.predict(test_features))
        else:
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

        y_train_r = train_labels_reg.reshape(-1, 1).astype(float)
        method_obj.fit(train_features, y_train_r, MSE,
                       epochs=args.max_iters, batch_size=args.batch_size,
                       learning_rate=args.lr)
        preds_train = method_obj.predict(train_features).squeeze()
        preds_test  = method_obj.predict(test_features).squeeze()

        loss_train = mse_fn(preds_train, train_labels_reg)
        loss_test  = mse_fn(preds_test,  test_labels_reg)

        print(f"\n[{args.method.upper()} | Regression]")
        print(f"  Train — MSE: {loss_train:.4f}")
        print(f"  {'Test ' if args.test else 'Val  '} — MSE: {loss_test:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification", type=str,
                        help="classification / regression")
    parser.add_argument("--method", default="dummy_classifier", type=str,
                        help="dummy_classifier / kmeans / mlp")
    parser.add_argument("--data_path", default="data/features.npz", type=str,
                        help="path to your dataset")
    parser.add_argument("--K", type=int, default=1,
                        help="number of clusters for kmeans")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the MLP")
    parser.add_argument("--max_iters", type=int, default=100,
                        help="max iters / epochs for iterative methods")
    parser.add_argument("--test", action="store_true",
                        help="evaluate on the test set instead of a validation set")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[128, 64],
                        help="hidden layer sizes for the MLP, e.g. --hidden_sizes 128 64")
    parser.add_argument("--activation", type=str, default="relu",
                        help="activation for the MLP: relu / sigmoid")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for MLP training")
    parser.add_argument("--weighted", action="store_true",
                        help="use class-weighted cross-entropy for the MLP (helps with imbalance)")

    args = parser.parse_args()
    main(args)