from sklearn.ensemble import RandomForestClassifier

from spine_segmentation.datasets.feature_dataset import FeatureDataModule


def main():
    feature_data = FeatureDataModule(batch_size=8000, sample_size=30)

    train_loader = feature_data.train_dataloader()
    X, Y = next(iter(train_loader))
    X = X.view(X.shape[0], -1)
    Y -= 1

    val_loader = feature_data.val_dataloader()
    X_val, Y_val = next(iter(val_loader))
    X_val = X_val.view(X_val.shape[0], -1)
    Y_val -= 1
    print("Train size", len(X), "Val size", len(X_val))

    # X, Y = [], []
    # for x, y in train_loader:
    #     X.append(x)
    #     Y.append(y)

    # Train a random forrest classifier
    # from sklearn.metrics import plot_confusion_matrix
    # from sklearn.metrics import plot_roc_curve
    # from sklearn.metrics import plot_precision_recall_curve
    # from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Use XGBoost
    # clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1000, random_state=0)
    clf = xgb.XGBClassifier()

    # Use random forest
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)

    clf.fit(X, Y)

    y_pred = clf.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))

    # Evaluate the model
    y_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(Y_val, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(Y_val, y_pred))
    print("Classification report:")
    print(classification_report(Y_val, y_pred))


if __name__ == "__main__":
    main()
