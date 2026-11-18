# Train on Google Colab

1. Use the supplied `pathsnap-source.zip`, or build it in PowerShell from the repository root.

   ```powershell
   Compress-Archive -Path backend/pathsnap,backend/requirements.txt,backend/requirements-colab.txt -DestinationPath pathsnap-source.zip -Force
   ```

2. Open [the training notebook](../training/PathSnap_Colab.ipynb) in Google Colab with a CPU runtime. Upload `pathsnap-source.zip` and the original `Student folder.zip` when the notebook prompts you. If the runtime still has the old five-model source ZIP, remove that ZIP from Colab's Files pane before running setup. You can reuse the dataset ZIP.
3. Run all cells. Download and extract `pathsnap-artifacts.zip` into `artifacts/` in this project. The archive includes `model.joblib`, the run results, trip inventory, comparison plots, and the student-trip map.

The full run fits seven classifiers on 80 training trips and evaluates on the ten student-test trips. The comparison includes Logistic Regression, Gaussian Naive Bayes, Random Forest, Extra Trees, Histogram Gradient Boosting, a multilayer perceptron (MLP), and XGBoost. Use the same three trip-grouped cross-validation folds for every model. The workflow chooses a model by mean CV candidate Macro F1 and runs a Bayesian search over that model's hyperparameters.

The exported decoder uses a 55 m transition scale, a disconnection penalty of 12, and a smoothed heading weight of 2. A short window around each GPS observation estimates direction; the heading contribution is applied when the window covers at least 15 m and at most 30 seconds. These settings were selected on training trips before the student-test evaluation. Results include tuned versus untuned comparisons and an evaluation with and without decoding. The provided route file contains an ordered arc sequence, so pointwise scores use labels derived by monotone GPS-to-reference-route alignment. Route continuity measures directly adjacent predicted arcs, and arc-overlap F1 is a multiset score.

## Read the experiment results

Open [the full Colab results](../artifacts/results.json). Each entry under `models` corresponds to one row in the comparison table.

| Report column | Field and interpretation |
| --- | --- |
| Configuration | `parameters` records the settings used for the untuned model. |
| CV Macro F1 | `mean_cv_macro_f1` averages the three grouped validation-fold scores and gives equal weight to the two candidate classes. |
| Student-test Top-1 | `student_test.top1_accuracy` is the share of GPS observations whose highest-scoring candidate matches the aligned reference arc. |
| Route continuity | `student_test.route_continuity` averages the share of consecutive predictions that repeat an arc or have directly connected endpoints for each trip. |
| Student-test Macro F1 | `student_test.macro_f1` gives equal weight to matching and non-matching candidate classification on student-test trips. |
| Log loss | `student_test.log_loss` measures the quality of candidate probabilities. Lower values are better. |
| Inference time | `student_test.inference_seconds` covers classifier evaluation across all ten test trips with features already prepared. API timing also includes file loading, feature generation, decoding, and scoring. |

Compare the selected model's original `student_test` entry with `tuned_student_test` to report the effect of Bayesian tuning. Use `decoder_comparison` to compare the tuned classifier alone with the tuned classifier plus HMM/Viterbi. Its `per_trip` entries show how the effect varies by trip.

Folium exports use the supplied road network as the background and do not request external map tiles. For fully offline HTML, run the visualisation module with `--leaflet-assets` pointing to the installed Leaflet distribution folder.
