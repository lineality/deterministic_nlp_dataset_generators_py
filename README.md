# deterministic_nlp_dataset_generators_py



```bash
$ cargo run --release -- --mode train   --data /home/oops/code/deterministic_nlp_dataset_generators_py/question_type_dataset_output/balanced_unified_question_type_dataset_v1.jsonl   --preset raw --engine byte-bag   --clauses 100 --vote-threshold 75 --states 85   --specificity 3.7 --vocab-size 3000 --ngram-len 8   --max-scan 4096 --epochs 4 --seed 128 --workers auto   --train-percent 80   --model-out /home/oops/models/unified_question_type_datasetv1.gmb   --log-out /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt
    Finished `release` profile [optimized] target(s) in 0.00s
     Running `target/release/ensemble_granmo --mode train --data /home/oops/code/deterministic_nlp_dataset_generators_py/question_type_dataset_output/balanced_unified_question_type_dataset_v1.jsonl --preset raw --engine byte-bag --clauses 100 --vote-threshold 75 --states 85 --specificity 3.7 --vocab-size 3000 --ngram-len 8 --max-scan 4096 --epochs 4 --seed 128 --workers auto --train-percent 80 --model-out /home/oops/models/unified_question_type_datasetv1.gmb --log-out /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt`
loaded 8040 labeled documents
resolved config: HarnessRunConfig { profile: PreprocessProfile { stage_bits: 0 }, engine_selection: ByteBag, patch_size: 5, stride: 2, bag_ngram_len: 8, bag_vocab_size: 3000, n_clauses: 100, vote_threshold: 75, states_per_action: 85, specificity: 3.7, max_scan_bytes: 4096, guarded_include: false, fire_guard_streak_limit: 0, epochs: 4, seed: 128, worker_count: 16 }

============================================================
               Classification Evaluation Report             
============================================================
  Run Preset:        raw          (Engine: byte-bag)
  Train/Test Split:  6432/1608 samples
Training Time Duration (h:m:s): 00:00:05
------------------------------------------------------------
  Accuracy (@ V > 0): 99.94%
  Best-F1 Threshold:  V > 1
  Precision:          1.0000
  Recall:             1.0000
  F1-Score:           1.0000
------------------------------------------------------------
Confusion Matrix (at optimal threshold):
                  Pred Neg (0)Pred Pos (1)
Actual Neg (0)    801         0           
Actual Pos (1)    0           807         
------------------------------------------------------------
Clause Dynamics:
  fire-rate over 1608 test docs: never 0/100  always 0/100 (0 vacuous, 0 specialized)  p25 12.7%  median 13.9%  p75 15.0%
  includes/clause: min 267  p25 475  median 539  p75 619  max 870  (0 clauses vacuous)
  vacuous vote offset: +0  (0 positive-polarity, 0 negative-polarity vacuous)
============================================================

misprediction log: appended 1 records to /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt
saved model artifact to /home/oops/models/unified_question_type_datasetv1.gmb

```



```bash
$ cargo run --release -- --mode train   --data /home/oops/code/deterministic_nlp_dataset_generators_py/question_type_dataset_output/balanced_unified_question_type_dataset_v2.jsonl  --preset raw --engine byte-bag   --clauses 100 --vote-threshold 75 --states 85   --specificity 3.7 --vocab-size 3000 --ngram-len 7   --max-scan 4096 --epochs 3 --seed 128 --workers auto   --train-percent 80   --model-out /home/oops/models/unified_question_type_datasetv1.gmb   --log-out /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt
    Finished `release` profile [optimized] target(s) in 0.00s
     Running `target/release/ensemble_granmo --mode train --data /home/oops/code/deterministic_nlp_dataset_generators_py/question_type_dataset_output/balanced_unified_question_type_dataset_v2.jsonl --preset raw --engine byte-bag --clauses 100 --vote-threshold 75 --states 85 --specificity 3.7 --vocab-size 3000 --ngram-len 7 --max-scan 4096 --epochs 3 --seed 128 --workers auto --train-percent 80 --model-out /home/oops/models/unified_question_type_datasetv1.gmb --log-out /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt`
loaded 199614 labeled documents
resolved config: HarnessRunConfig { profile: PreprocessProfile { stage_bits: 0 }, engine_selection: ByteBag, patch_size: 5, stride: 2, bag_ngram_len: 7, bag_vocab_size: 3000, n_clauses: 100, vote_threshold: 75, states_per_action: 85, specificity: 3.7, max_scan_bytes: 4096, guarded_include: false, fire_guard_streak_limit: 0, epochs: 3, seed: 128, worker_count: 16 }

============================================================
               Classification Evaluation Report             
============================================================
  Run Preset:        raw          (Engine: byte-bag)
  Train/Test Split:  159691/39923 samples
Training Time Duration (h:m:s): 00:01:48
------------------------------------------------------------
  Accuracy (@ V > 0): 99.47%
  Best-F1 Threshold:  V > 0
  Precision:          0.9997
  Recall:             0.9896
  F1-Score:           0.9946
------------------------------------------------------------
Confusion Matrix (at optimal threshold):
                  Pred Neg (0)Pred Pos (1)
Actual Neg (0)    20081       5           
Actual Pos (1)    207         19630       
------------------------------------------------------------
Clause Dynamics:
  fire-rate over 39923 test docs: never 0/100  always 0/100 (0 vacuous, 0 specialized)  p25 14.9%  median 15.6%  p75 16.3%
  includes/clause: min 450  p25 622  median 719  p75 797  max 1064  (0 clauses vacuous)
  vacuous vote offset: +0  (0 positive-polarity, 0 negative-polarity vacuous)
============================================================

misprediction log: appended 212 records to /home/oops/code/granmo_model_nlp_classifier_rust/para_byte_ganmo/logs/trash.txt
saved model artifact to /home/oops/models/unified_question_type_datasetv1.gmb
```
