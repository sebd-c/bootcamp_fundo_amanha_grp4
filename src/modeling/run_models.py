# imports
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from src.models.df_based._data_split import get_data_split
from src.models.df_based._train import (build_param_grid,
                                        build_model,
                                        get_select_k_best_features,
                                        run_cross_validation,
                                        save_model
                                        )
from src.plotters._get_plots import (plot_cv_metrics,
                                     plot_feature_summary_heatmap,
                                     plot_select_k_best_scores)
from src.plotters._get_interpret   import (run_permutation_importance,
                                           plot_permutation_importance,
                                           run_ceteris_paribus,
                                           run_ice_plots,
                                           get_shap_explainer,
                                           run_shap_analysis)
import dalex as dx

#################################################################################
# global vars of relative paths
INPUT_PATH = '/home/debs/data/hetsencelldb/csv_tables/cleaned_morphometry_labeled.csv'
OUTPUT_DIR = '/home/debs/data/hetsencelldb/tab_model_outputs'

#################################################################################
def run_pipeline(X_train,
                 y_train,
                 X_test,
                 y_test,
                 # search config
                 param_grid: dict | None = None,
                 model_name: str = "dt",
                 search_scoring: str = "f1_macro",
                 cv_inner: int = 5,
                 # outer CV config
                 cv_scoring: dict | None = None,
                 n_splits: int = 5,
                 n_repeats: int = 3,
                 cv_n_jobs: int = -1,
                 # output paths
                 output_dir: str = OUTPUT_DIR,
                 model_prefix: str = "decision_tree",
                 plot_format: str = "png",
                 # misc
                 random_state: int = 42,
                 ) -> dict:
    """
    Whole pipeline function from hyperparameter tuning
    to model fitting in whole dataset
    """

    print("\n══ Step 1 / 8 — Building GridSearchCV model ══")
    model = build_model(model= model_name,
                        param_grid= param_grid,
                        scoring= search_scoring,
                        cv_inner= cv_inner,
                        random_state = random_state,
                        )

    print("\n══ Step 2 / 8 — Running outer cross-validation ══")
    scores_df = run_cross_validation(model= model,
                                     X= X_train,
                                     y= y_train,
                                     n_splits= n_splits,
                                     n_repeats= n_repeats,
                                     random_state= random_state,
                                     scoring= cv_scoring,
                                     n_jobs= cv_n_jobs,
                                     )
    # save it as .csv
    metrics_filename = model_prefix + "_cv_metrics.csv"
    metrics_output_path = os.path.join(output_dir, metrics_filename)
    scores_df.to_csv(metrics_output_path, index=False)
    print(scores_df.describe().T[["mean", "std"]].round(4))

    print("\n══ Step 3 / 8 — Plotting CV metrics ══")
    plot_cv_metrics(scores_df,
                    output_dir=output_dir,
                    file_format=plot_format,
                    model_prefix=model_prefix
                    )

    print("\n══ Step 4 / 8 — Fitting on full data & saving model ══")
    model_filename = model_prefix + "_model.joblib"
    model_path   = os.path.join(output_dir, model_filename)
    fitted_model = save_model(model, X_train, y_train, model_path=model_path)
    print(f"\nFinal full-training refit params:\n{fitted_model.best_params_}\n")

    params_df = pd.DataFrame([fitted_model.best_params_])
    params_df.insert(0, "model", model_prefix)
    model_hyperparam_filename = model_prefix + "_fitted_hyperparams.csv"
    model_hyperparam_output_path = os.path.join(output_dir, model_hyperparam_filename)
    params_df.to_csv(model_hyperparam_output_path, index=False)
    print(f"Fitted model params saved @ {model_hyperparam_output_path}")

    selected_features = get_select_k_best_features(fitted_model=fitted_model,
                                                   feature_names=X_train.columns)
    selected_features_filename = model_prefix + "_selected_features.csv"
    selected_features_path = os.path.join(output_dir, selected_features_filename)
    selected_features.to_csv(selected_features_path, index=False)
    print(f"\nSelected features saved into {selected_features_path}")

    plot_select_k_best_scores(features_df=selected_features,
                              output_dir=output_dir,
                              file_format=plot_format,
                              model_prefix=model_prefix,
                              )

    if model_name == "tabicl":
        print("\n══ Step 5 / 8 — Skipping permutation importance for TabICL ══")
    else:
        print("\n══ Step 5 / 8 — permutation importance ══")
        perm_imp_csv_filename = model_prefix + "permutation_importance.csv"
        perm_imp_csv_output_path = os.path.join(output_dir, perm_imp_csv_filename)
        perm_imp_png_filename = model_prefix + "permutation_importance.png"
        perm_imp_png_output_path = os.path.join(output_dir, perm_imp_png_filename)
        perm_df = run_permutation_importance(clf=fitted_model,
                                             X_train=X_train,
                                             y_train=y_train,
                                             feature_names=X_train.columns.tolist(),
                                             output_path=perm_imp_csv_output_path,
                                             )
        plot_permutation_importance(perm_df,
                                    output_path=perm_imp_png_output_path,
                                    model_prefix=model_prefix)

    print("\n══ Step x / 8 — plotting for test ══")
    # class names
    class_names = ['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']

    y_pred = fitted_model.predict(X_test)

    cf_img_png_filename = model_prefix + "confusion_matrix.png"
    cf_imp_png_output_path = os.path.join(output_dir, cf_img_png_filename)
    # confusion matrix
    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, normalize='true')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title(f'Classification Confusion Matrix Normalized Test ({model_prefix})')
    plt.tight_layout()

    fig.savefig(cf_imp_png_output_path)
    plt.close(fig)
    #TODO: fix ceteris paribus
    # print("\n══ Step 6 / 8 — ceteris paribus ══")
    # exp = dx.Explainer(fitted_model, X_train, y_train, label=model_prefix)
    # cet_pari_csv_filename = model_prefix + "ceteris_paribus.csv"
    # cet_pari_csv_output_path = os.path.join(output_dir, cet_pari_csv_filename)
    # cp_df = run_ceteris_paribus(exp=exp,
    #                             X=X_train,
    #                             y=y_train,
    #                             observation_indices=[2, 5, 10],
    #                             output_dir=f"{OUTPUT_DIR}/cp_profiles",
    #                             combined_path=cet_pari_csv_output_path,
    #                             )

    # print("\n══ Step 7 / 8 — ICE + PDP ══")
    # run_ice_plots(model=fitted_model,
    #               X_train=X_train,
    #               feature_list=selected_features["feature"].tolist(),
    #               output_dir=f"{OUTPUT_DIR}/ice_plots",
    #               subsample=0.1,
    #               random_state=42,
    #               model_prefix=model_prefix
    #               )
    #
    # # Step 8 — SHAP
    # shap_df = run_shap_analysis(clf=fitted_model,
    #                             X_train=X_train,
    #                             observation_indices=[2, 5, 10],
    #                             selected_features=selected_features,
    #                             output_dir=f"{OUTPUT_DIR}/shap_plots",
    #                             model_prefix=model_prefix
    #                             )

    print("\n══ Pipeline complete ══\n")

    return {"model": fitted_model,
            "cv_scores": scores_df,
            "selected_features": selected_features,
            "model_path": model_path,
            }


####################################################
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # read df
    data = pd.read_csv(INPUT_PATH)

    # split data
    X_train, X_test, y_train, y_test = get_data_split(data, seed=42)

    # save used splits
    X_train_filename = 'x_train_data.csv'
    X_train_saved_path = os.path.join(OUTPUT_DIR, X_train_filename)
    X_train.to_csv(X_train_saved_path, index=False)

    X_test_filename = 'X_test_data.csv'
    X_test_saved_path = os.path.join(OUTPUT_DIR, X_test_filename)
    X_test.to_csv(X_test_saved_path, index=False)

    y_train_filename = 'y_train_data.csv'
    y_train_saved_path = os.path.join(OUTPUT_DIR, y_train_filename)
    y_train.to_csv(y_train_saved_path, index=False)

    y_test_filename = 'y_test_data.csv'
    y_test_saved_path = os.path.join(OUTPUT_DIR, y_test_filename)
    y_test.to_csv(y_test_saved_path, index=False)

    # make param grids by models
    param_grid_dt = build_param_grid(model= "decision_tree")
    param_grid_rf = build_param_grid(model= "random_forest")
    param_grid_knn = build_param_grid(model= "knn")
    param_grid_svc = build_param_grid(model= "svc")
    # param_grid_nb = build_param_grid(model= "naive_bayes")
    param_grid_xgb = build_param_grid(model= "xgboost")
    param_grid_tabicl = build_param_grid(model="tabicl")

    # plot heatmatp feats
    plot_feature_summary_heatmap(df= X_train,
                                 output_dir=OUTPUT_DIR,
                                 file_format='png')
    # training process
    # decision tree
    results_dt = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              X_test=X_test,
                              y_test=y_test,
                              param_grid= param_grid_dt,
                              model_name="dt",
                              cv_inner= 5,
                              n_splits=5,
                              n_repeats=3,
                              output_dir=OUTPUT_DIR,
                              model_prefix="decision_tree",
                              random_state = 42,
                              )

    # random forest
    results_rf = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              X_test=X_test,
                              y_test=y_test,
                              param_grid=param_grid_rf,
                              model_name="rf",
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=3,
                              output_dir=OUTPUT_DIR,
                              model_prefix="random_forest",
                              random_state=42,
                              )

    # knn
    results_knn = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               X_test=X_test,
                               y_test=y_test,
                               param_grid=param_grid_knn,
                               model_name="knn",
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=3,
                               output_dir=OUTPUT_DIR,
                               model_prefix="knn",
                               random_state=42,
                               )

    # naive bayes
    results_svc = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              X_test=X_test,
                              y_test=y_test,
                              param_grid=param_grid_svc,
                              model_name="svc",
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=3,
                              output_dir=OUTPUT_DIR,
                              model_prefix="svc",
                              random_state=42,
                              )

    # xgboost
    results_xgb = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               X_test=X_test,
                               y_test=y_test,
                               param_grid=param_grid_xgb,
                               model_name="xgb",
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=3,
                               output_dir=OUTPUT_DIR,
                               model_prefix="xgboost",
                               random_state=42,
                               )

    # TabICL pretrained tabular foundation model
    results_tabicl = run_pipeline(X_train=X_train,
                                  y_train=y_train,
                                  X_test=X_test,
                                  y_test=y_test,
                                  param_grid=param_grid_tabicl,
                                  model_name="tabicl",
                                  search_scoring="f1_macro",
                                  cv_inner=5,
                                  n_splits=5,
                                  n_repeats=3,
                                  cv_n_jobs=1,
                                  output_dir=OUTPUT_DIR,
                                  model_prefix="tabicl",
                                  random_state=42,
                                  )

    # Evaluate on held-out test set,
    # only after model tuning
    # fitted_model = results["model"]
    # preds = predict_new_data(fitted_model, X_test, proba=True)
    # print("Test predictions (first 10):", preds["predictions"][:10])
    # print("Test accuracy:", accuracy_score(y_test, preds["predictions"]).round(4))

    # ── Later: load and reuse ──────────────────
    # model = load_model("outputs/decision_tree_model.joblib")
    # preds = predict_new_data(model, X_new)
