# %% [markdown]
# ---
# title: Can one prediction target be reset and relearned?
# categories:
# - Schema mutation
# evidence-id: P048
# description: Reset the learned predictor for u while preserving its sibling target v.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Reset target, before retraining
#   metric:
#   - arms
#   - selective_reset
#   - before
#   - u
#   - nrmse
#   format: error
# - label: Sibling immediately after reset
#   metric:
#   - arms
#   - selective_reset
#   - before
#   - v
#   - nrmse
#   format: error
# - label: Reset target, after retraining
#   metric:
#   - arms
#   - selective_reset
#   - after
#   - u
#   - nrmse
#   format: error
# evidence-group: Mutations
# evidence-result:
#   accelerator: cpu
#   checks:
#     Complete reset loses both learned tasks: true
#     Hidden target placeholders cannot affect predictions: true
#     Relearned checkpoint preserves predictions: true
#     Relearned checkpoint preserves schema and state: true
#     Reset changes selected learned state: true
#     Reset head parameters actually relearn: true
#     Reset preserves every unselected state entry: true
#     Reset preserves the schema: true
#     Selected target immediately loses its skill: true
#     Source learns both targets below 0.25 nRMSE: true
#     Unselected target immediately preserves predictions: true
#     'complete_reset: both final targets below 0.25 nRMSE': true
#     'complete_reset: fresh optimizer covers current parameters': true
#     'continuation: both final targets below 0.25 nRMSE': true
#     'continuation: fresh optimizer covers current parameters': true
#     'selective_reset: both final targets below 0.25 nRMSE': true
#     'selective_reset: fresh optimizer covers current parameters': true
#   duration_seconds: 41.977
#   metrics:
#     arms:
#       complete_reset:
#         after:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 0.021974634224565144
#             rmse: 0.028776592197552985
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 0.04058309755202279
#             rmse: 0.05397550940714712
#         before:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 1.0754545900036776
#             rmse: 1.408347363021229
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 1.076282389877726
#             rmse: 1.4314553043942415
#         changed_state_entries:
#         - nodes./a.embedder.embeddings.weight
#         - nodes./a.embedder.counter.counts
#         - nodes./a.embedder.linear.weight
#         - nodes./a.embedder.linear.bias
#         - nodes./a.embedder.scalar.weight
#         - nodes./a.embedder.normalizers.raw.mean
#         - nodes./a.embedder.normalizers.raw.var
#         - nodes./a.embedder.normalizers.raw.count
#         - nodes./a.embedder.normalizers.log.mean
#         - nodes./a.embedder.normalizers.log.var
#         - nodes./a.embedder.normalizers.log.count
#         - nodes./b.embedder.embeddings.weight
#         - nodes./b.embedder.counter.counts
#         - nodes./b.embedder.linear.weight
#         - nodes./b.embedder.linear.bias
#         - nodes./b.embedder.scalar.weight
#         - nodes./b.embedder.normalizers.raw.mean
#         - nodes./b.embedder.normalizers.raw.var
#         - nodes./b.embedder.normalizers.raw.count
#         - nodes./b.embedder.normalizers.log.mean
#         - nodes./b.embedder.normalizers.log.var
#         - nodes./b.embedder.normalizers.log.count
#         - nodes./u.embedder.embeddings.weight
#         - nodes./u.embedder.counter.counts
#         - nodes./u.embedder.linear.weight
#         - nodes./u.embedder.linear.bias
#         - nodes./u.embedder.scalar.weight
#         - nodes./u.embedder.normalizers.raw.mean
#         - nodes./u.embedder.normalizers.raw.var
#         - nodes./u.embedder.normalizers.raw.count
#         - nodes./u.embedder.normalizers.log.mean
#         - nodes./u.embedder.normalizers.log.var
#         - nodes./u.embedder.normalizers.log.count
#         - nodes./u.decoder.context_projection.0.weight
#         - nodes./u.decoder.context_projection.1.weight
#         - nodes./u.decoder.context_projection.1.bias
#         - nodes./u.decoder.context_projection.3.weight
#         - nodes./u.decoder.context_projection.3.bias
#         - nodes./u.decoder.pool.queries
#         - nodes./u.decoder.pool.blocks.0.attention_norm.weight
#         - nodes./u.decoder.pool.blocks.0.ffn_norm.weight
#         - nodes./u.decoder.pool.blocks.0.attention.head_scale
#         - nodes./u.decoder.pool.blocks.0.attention.q_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.q_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.k_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.k_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.v_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.v_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.out_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.out_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.q_norm.weight
#         - nodes./u.decoder.pool.blocks.0.attention.k_norm.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.0.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.0.bias
#         - nodes./u.decoder.pool.blocks.0.ffn.3.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.3.bias
#         - nodes./u.decoder.pool.norm.weight
#         - nodes./u.decoder.classification.weight
#         - nodes./u.decoder.classification.bias
#         - nodes./u.decoder.regression.weight
#         - nodes./u.decoder.regression.bias
#         - nodes./u.decoder.normalizer.mean
#         - nodes./u.decoder.normalizer.var
#         - nodes./u.decoder.normalizer.count
#         - nodes./v.embedder.embeddings.weight
#         - nodes./v.embedder.counter.counts
#         - nodes./v.embedder.linear.weight
#         - nodes./v.embedder.linear.bias
#         - nodes./v.embedder.scalar.weight
#         - nodes./v.embedder.normalizers.raw.mean
#         - nodes./v.embedder.normalizers.raw.var
#         - nodes./v.embedder.normalizers.raw.count
#         - nodes./v.embedder.normalizers.log.mean
#         - nodes./v.embedder.normalizers.log.var
#         - nodes./v.embedder.normalizers.log.count
#         - nodes./v.decoder.context_projection.0.weight
#         - nodes./v.decoder.context_projection.1.weight
#         - nodes./v.decoder.context_projection.1.bias
#         - nodes./v.decoder.context_projection.3.weight
#         - nodes./v.decoder.context_projection.3.bias
#         - nodes./v.decoder.pool.queries
#         - nodes./v.decoder.pool.blocks.0.attention_norm.weight
#         - nodes./v.decoder.pool.blocks.0.ffn_norm.weight
#         - nodes./v.decoder.pool.blocks.0.attention.head_scale
#         - nodes./v.decoder.pool.blocks.0.attention.q_proj.weight
#         - nodes./v.decoder.pool.blocks.0.attention.q_proj.bias
#         - nodes./v.decoder.pool.blocks.0.attention.k_proj.weight
#         - nodes./v.decoder.pool.blocks.0.attention.k_proj.bias
#         - nodes./v.decoder.pool.blocks.0.attention.v_proj.weight
#         - nodes./v.decoder.pool.blocks.0.attention.v_proj.bias
#         - nodes./v.decoder.pool.blocks.0.attention.out_proj.weight
#         - nodes./v.decoder.pool.blocks.0.attention.out_proj.bias
#         - nodes./v.decoder.pool.blocks.0.attention.q_norm.weight
#         - nodes./v.decoder.pool.blocks.0.attention.k_norm.weight
#         - nodes./v.decoder.pool.blocks.0.ffn.0.weight
#         - nodes./v.decoder.pool.blocks.0.ffn.0.bias
#         - nodes./v.decoder.pool.blocks.0.ffn.3.weight
#         - nodes./v.decoder.pool.blocks.0.ffn.3.bias
#         - nodes./v.decoder.pool.norm.weight
#         - nodes./v.decoder.classification.weight
#         - nodes./v.decoder.classification.bias
#         - nodes./v.decoder.regression.weight
#         - nodes./v.decoder.regression.bias
#         - nodes./v.decoder.normalizer.mean
#         - nodes./v.decoder.normalizer.var
#         - nodes./v.decoder.normalizer.count
#         - nodes./.encoder.encoder.0.attention_norm.weight
#         - nodes./.encoder.encoder.0.ffn_norm.weight
#         - nodes./.encoder.encoder.0.attention.head_scale
#         - nodes./.encoder.encoder.0.attention.q_proj.weight
#         - nodes./.encoder.encoder.0.attention.q_proj.bias
#         - nodes./.encoder.encoder.0.attention.k_proj.weight
#         - nodes./.encoder.encoder.0.attention.k_proj.bias
#         - nodes./.encoder.encoder.0.attention.v_proj.weight
#         - nodes./.encoder.encoder.0.attention.v_proj.bias
#         - nodes./.encoder.encoder.0.attention.out_proj.weight
#         - nodes./.encoder.encoder.0.attention.out_proj.bias
#         - nodes./.encoder.encoder.0.attention.q_norm.weight
#         - nodes./.encoder.encoder.0.attention.k_norm.weight
#         - nodes./.encoder.encoder.0.ffn.0.weight
#         - nodes./.encoder.encoder.0.ffn.0.bias
#         - nodes./.encoder.encoder.0.ffn.3.weight
#         - nodes./.encoder.encoder.0.ffn.3.bias
#         - nodes./.encoder.coordinate_encoder.attention_norm.weight
#         - nodes./.encoder.coordinate_encoder.ffn_norm.weight
#         - nodes./.encoder.coordinate_encoder.attention.head_scale
#         - nodes./.encoder.coordinate_encoder.attention.q_proj.weight
#         - nodes./.encoder.coordinate_encoder.attention.q_proj.bias
#         - nodes./.encoder.coordinate_encoder.attention.k_proj.weight
#         - nodes./.encoder.coordinate_encoder.attention.k_proj.bias
#         - nodes./.encoder.coordinate_encoder.attention.v_proj.weight
#         - nodes./.encoder.coordinate_encoder.attention.v_proj.bias
#         - nodes./.encoder.coordinate_encoder.attention.out_proj.weight
#         - nodes./.encoder.coordinate_encoder.attention.out_proj.bias
#         - nodes./.encoder.coordinate_encoder.attention.q_norm.weight
#         - nodes./.encoder.coordinate_encoder.attention.k_norm.weight
#         - nodes./.encoder.coordinate_encoder.ffn.0.weight
#         - nodes./.encoder.coordinate_encoder.ffn.0.bias
#         - nodes./.encoder.coordinate_encoder.ffn.3.weight
#         - nodes./.encoder.coordinate_encoder.ffn.3.bias
#         - nodes./.encoder.pool.queries
#         - nodes./.encoder.pool.mass_direction
#         - nodes./.encoder.pool.blocks.0.attention_norm.weight
#         - nodes./.encoder.pool.blocks.0.ffn_norm.weight
#         - nodes./.encoder.pool.blocks.0.attention.head_scale
#         - nodes./.encoder.pool.blocks.0.attention.q_proj.weight
#         - nodes./.encoder.pool.blocks.0.attention.q_proj.bias
#         - nodes./.encoder.pool.blocks.0.attention.k_proj.weight
#         - nodes./.encoder.pool.blocks.0.attention.k_proj.bias
#         - nodes./.encoder.pool.blocks.0.attention.v_proj.weight
#         - nodes./.encoder.pool.blocks.0.attention.v_proj.bias
#         - nodes./.encoder.pool.blocks.0.attention.out_proj.weight
#         - nodes./.encoder.pool.blocks.0.attention.out_proj.bias
#         - nodes./.encoder.pool.blocks.0.attention.q_norm.weight
#         - nodes./.encoder.pool.blocks.0.attention.k_norm.weight
#         - nodes./.encoder.pool.blocks.0.ffn.0.weight
#         - nodes./.encoder.pool.blocks.0.ffn.0.bias
#         - nodes./.encoder.pool.blocks.0.ffn.3.weight
#         - nodes./.encoder.pool.blocks.0.ffn.3.bias
#         - nodes./.encoder.pool.norm.weight
#         - nodes./.encoder.pool.mass_projection.weight
#         initial_prediction_drift:
#           u: 3.2319892467738436
#           v: 3.4135874313300847
#         optimizer_covers_current_parameters: true
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 1.0717364556897648
#               rmse: 1.4167276343671031
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 1.0915005748356803
#               rmse: 1.4204260031786007
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.042266329488635034
#               rmse: 0.05587182993721719
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.11351702289735092
#               rmse: 0.1477255576810778
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.03801136996607654
#               rmse: 0.050247202066513286
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.03282601049766878
#               rmse: 0.04271818079300773
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.022688586462311863
#               rmse: 0.029992025796301877
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.04058621802789518
#               rmse: 0.052816939163020854
#           step: 256
#       continuation:
#         after:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 0.009317648261764334
#             rmse: 0.01220180329414956
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 0.013346120695056361
#             rmse: 0.0177503371249943
#         before:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 0.03906345643191986
#             rmse: 0.05115503375758606
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 0.021785454454026798
#             rmse: 0.028974648874816825
#         changed_state_entries: []
#         initial_prediction_drift:
#           u: 0.0
#           v: 0.0
#         optimizer_covers_current_parameters: true
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.038480521940841346
#               rmse: 0.050867373717704736
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.021636498490811007
#               rmse: 0.02815669160660699
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.05953565449582029
#               rmse: 0.07870013799248443
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.07286833217002874
#               rmse: 0.09482731957163336
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.011675550315067326
#               rmse: 0.015433901394306549
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.007781872252997481
#               rmse: 0.010126951791331601
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.00944980231037823
#               rmse: 0.01249168674009751
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.01330915628352485
#               rmse: 0.01731989162564808
#           step: 256
#       selective_reset:
#         after:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 0.0611153532767438
#             rmse: 0.0800328041996814
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 0.015086485446637432
#             rmse: 0.02006502180130357
#         before:
#           u:
#             baseline_rmse: 1.3095368006346164
#             nrmse: 1.1079495613817283
#             rmse: 1.4509007238763552
#           v:
#             baseline_rmse: 1.3299997452869836
#             nrmse: 0.021785454454026798
#             rmse: 0.028974648874816825
#         changed_state_entries:
#         - nodes./u.embedder.embeddings.weight
#         - nodes./u.embedder.counter.counts
#         - nodes./u.embedder.linear.weight
#         - nodes./u.embedder.linear.bias
#         - nodes./u.embedder.scalar.weight
#         - nodes./u.embedder.normalizers.raw.mean
#         - nodes./u.embedder.normalizers.raw.var
#         - nodes./u.embedder.normalizers.raw.count
#         - nodes./u.embedder.normalizers.log.mean
#         - nodes./u.embedder.normalizers.log.var
#         - nodes./u.embedder.normalizers.log.count
#         - nodes./u.decoder.context_projection.0.weight
#         - nodes./u.decoder.context_projection.1.weight
#         - nodes./u.decoder.context_projection.1.bias
#         - nodes./u.decoder.context_projection.3.weight
#         - nodes./u.decoder.context_projection.3.bias
#         - nodes./u.decoder.pool.queries
#         - nodes./u.decoder.pool.blocks.0.attention_norm.weight
#         - nodes./u.decoder.pool.blocks.0.ffn_norm.weight
#         - nodes./u.decoder.pool.blocks.0.attention.head_scale
#         - nodes./u.decoder.pool.blocks.0.attention.q_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.q_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.k_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.k_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.v_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.v_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.out_proj.weight
#         - nodes./u.decoder.pool.blocks.0.attention.out_proj.bias
#         - nodes./u.decoder.pool.blocks.0.attention.q_norm.weight
#         - nodes./u.decoder.pool.blocks.0.attention.k_norm.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.0.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.0.bias
#         - nodes./u.decoder.pool.blocks.0.ffn.3.weight
#         - nodes./u.decoder.pool.blocks.0.ffn.3.bias
#         - nodes./u.decoder.pool.norm.weight
#         - nodes./u.decoder.classification.weight
#         - nodes./u.decoder.classification.bias
#         - nodes./u.decoder.regression.weight
#         - nodes./u.decoder.regression.bias
#         - nodes./u.decoder.normalizer.mean
#         - nodes./u.decoder.normalizer.var
#         - nodes./u.decoder.normalizer.count
#         initial_prediction_drift:
#           u: 3.297214376243461
#           v: 0.0
#         optimizer_covers_current_parameters: true
#         reset_head_updated_parameters:
#         - embedder.embeddings.weight
#         - embedder.linear.weight
#         - embedder.scalar.weight
#         - decoder.context_projection.0.weight
#         - decoder.context_projection.1.weight
#         - decoder.context_projection.1.bias
#         - decoder.context_projection.3.weight
#         - decoder.context_projection.3.bias
#         - decoder.pool.queries
#         - decoder.pool.blocks.0.attention_norm.weight
#         - decoder.pool.blocks.0.ffn_norm.weight
#         - decoder.pool.blocks.0.attention.head_scale
#         - decoder.pool.blocks.0.attention.q_proj.weight
#         - decoder.pool.blocks.0.attention.q_proj.bias
#         - decoder.pool.blocks.0.attention.k_proj.weight
#         - decoder.pool.blocks.0.attention.k_proj.bias
#         - decoder.pool.blocks.0.attention.v_proj.weight
#         - decoder.pool.blocks.0.attention.v_proj.bias
#         - decoder.pool.blocks.0.attention.out_proj.weight
#         - decoder.pool.blocks.0.attention.out_proj.bias
#         - decoder.pool.blocks.0.attention.q_norm.weight
#         - decoder.pool.blocks.0.attention.k_norm.weight
#         - decoder.pool.blocks.0.ffn.0.weight
#         - decoder.pool.blocks.0.ffn.0.bias
#         - decoder.pool.blocks.0.ffn.3.weight
#         - decoder.pool.blocks.0.ffn.3.bias
#         - decoder.pool.norm.weight
#         - decoder.classification.weight
#         - decoder.classification.bias
#         - decoder.regression.weight
#         - decoder.regression.bias
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 1.0969391621473115
#               rmse: 1.4500430735403174
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.021636498490811007
#               rmse: 0.02815669160660699
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.04596460903108004
#               rmse: 0.06076058295068473
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.07448751366000878
#               rmse: 0.09693444397015176
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.01737968206476202
#               rmse: 0.022974188968701154
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.00993420707741747
#               rmse: 0.01292789612671434
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3218992662289384
#               nrmse: 0.059823203206791825
#               rmse: 0.07908024842252279
#             v:
#               baseline_rmse: 1.3013515850804187
#               nrmse: 0.015615807338391338
#               rmse: 0.020321655632126002
#           step: 256
#     baseline: Source-training mean for test comparisons; phase-training mean for validation curves
#     downstream_interpretable: true
#     mutation: reset /u; complete-reset control resets record with descendants=True
#     optimizer_policy: New AdamW factory and Trainer for every fit; both hidden labels rehearsed
#     source:
#       u:
#         baseline_rmse: 1.3095368006346164
#         nrmse: 0.03906345643191986
#         rmse: 0.05115503375758606
#       v:
#         baseline_rmse: 1.3299997452869836
#         nrmse: 0.021785454454026798
#         rmse: 0.028974648874816825
#     source_fit:
#       optimizer_covers_current_parameters: true
#       steps: 512
#       validation_curve:
#       - scores:
#           u:
#             baseline_rmse: 1.3363771131216209
#             nrmse: 0.9802431086852679
#             rmse: 1.3099744557421815
#           v:
#             baseline_rmse: 1.2518777806823944
#             nrmse: 1.3070605619809677
#             rmse: 1.636280075550217
#         step: 0
#       - scores:
#           u:
#             baseline_rmse: 1.3363771131216209
#             nrmse: 0.048544761465038895
#             rmse: 0.06487410818382638
#           v:
#             baseline_rmse: 1.2518777806823944
#             nrmse: 0.05565530129411444
#             rmse: 0.06967363506728598
#         step: 32
#       - scores:
#           u:
#             baseline_rmse: 1.3363771131216209
#             nrmse: 0.012703294783821564
#             rmse: 0.016976392410336407
#           v:
#             baseline_rmse: 1.2518777806823944
#             nrmse: 0.018296239865371232
#             rmse: 0.02290465615749369
#         step: 128
#       - scores:
#           u:
#             baseline_rmse: 1.3363771131216209
#             nrmse: 0.03879393358536631
#             rmse: 0.05184332497144372
#           v:
#             baseline_rmse: 1.2518777806823944
#             nrmse: 0.022948694070572005
#             rmse: 0.028728960202626906
#         step: 512
#     source_prerequisite_met: true
#     test_rows: 1024
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 38ba15dca45ae4193572bbdffc2e9102f836a9f131de47a6567891b108f331b4
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: ea5378710bbc
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: c724f4f4d48b06bfd36641dc9167c474323779e9b30c13a497598b911b8f63bf
#     source_sha256: 27346a47ec16358e843cdad75e15d224edd096930b4d6c7cdb2a22a7bfa39228
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 4801
#   started_at: '2026-11-01T01:46:32.404496+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# a: 0.5
# b: 0.25
# u: 1.0
# v: 0.75
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mutations-reset-target
# //| fig-alt: "Model tree with record, a, b, u, v. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number"),
#   node("b", type: "Number"),
#   node("u", type: "Number", kind: "target", body: [Reset, then relearned]),
#   node("v", type: "Number", kind: "target", body: [Unselected, always hidden]),
# )))
# ```
#
# ## Comparison
#
# Measure both immediately after reset and after further training on both targets. Compare unchanged and completely reset models.
#
# ## Result
#
# {{< evidence P048 status >}}
#
# The selected target loses its learned prediction and then relearns; its sibling survives the reset. Resetting a shared branch is a different case.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P048 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P048 script >}}
#
# ### Remaining work
#
# Three GPU seeds support this case; calibrate gates on ten separate seeds. Resetting a
# context-producing branch, resetting descendants, and relearning without
# sibling labels require their own experiments. Localized loss here depends
# on the selected field being hidden from the shared context.
#
# ### Complete experiment code
#

# %%
"""P048: localize loss of learned behavior after reset and demonstrate relearning."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P048"
TARGETS = ("u", "v")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "u": a + 2 * b, "v": 2 * a - b}


def prediction(model: rf.Model, rows: list[dict], *, corrupt: bool = False) -> dict:
    inputs = [{"a": row["a"], "b": row["b"]} for row in rows]
    if corrupt:
        for row in inputs:
            row.update({name: 1000.0 for name in TARGETS})
    output = model.predict(inputs)["predictions"].to_pylist()
    return {name: np.asarray([row[name]["content"] for row in output], dtype=np.float64) for name in TARGETS}


def scores(rows: list[dict], predicted: dict, means: dict) -> dict:
    measured = {}
    for name, values in predicted.items():
        actual = np.asarray([row[name] for row in rows])
        baseline = float(np.sqrt(np.mean((actual - means[name]) ** 2)))
        rmse = float(np.sqrt(np.mean((actual - values) ** 2)))
        measured[name] = {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}
    return measured


def equal(first, second) -> bool:
    """Compare tensor and extension-owned state, including normalization."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[name]) for name, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


class Curve(lit.Callback):
    """Observe fixed validation checkpoints without restarting the optimizer."""

    def __init__(self, rows: list[dict], means: dict, budget: int):
        self.rows, self.means = rows, means
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append(
            {
                "step": 0,
                "scores": scores(self.rows, prediction(pl_module, self.rows), self.means),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {
                    "step": trainer.global_step,
                    "scores": scores(self.rows, prediction(pl_module, self.rows), self.means),
                }
            )


def fit(model: rf.Model, *, seed: int, split: int, budget: int, accelerator: str) -> dict:
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=split))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    curve = Curve(list(records(rows=512, seed=split + 1)), means, budget)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=split),
        validate=partial(records, rows=512, seed=split + 1),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.003,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        callbacks=[curve],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    optimized = {id(p) for group in trainer.optimizers[0].param_groups for p in group["params"]}
    current = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "steps": trainer.global_step,
        "optimizer_covers_current_parameters": current == optimized,
        "validation_curve": curve.measurements,
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    source_budget = 512 if steps is None else min(steps, 512)
    adapt_budget = 256 if steps is None else min(steps, 256)
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    scales = {name: float(np.std([row[name] for row in training])) for name in TARGETS}
    source = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        a=rf.Number,
        b=rf.Number,
        u=rf.Number(mask=True, objective="mse"),
        v=rf.Number(mask=True, objective="mse"),
    )
    source_fit = fit(source, seed=seed, split=seed + 1, budget=source_budget, accelerator=accelerator)
    reference = prediction(source, test)
    initial = scores(test, reference, means)
    learned = all(value["nrmse"] < 0.25 for value in initial.values())
    checks = {"Source learns both targets below 0.25 nRMSE": learned}
    state = deepcopy(source.state_dict())
    arms = {}

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("selective_reset", "continuation", "complete_reset")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = rf.Model.load(checkpoint).to(source.device)
            if arm == "selective_reset":
                model.reset(rf.where("address") == "/u")
            elif arm == "complete_reset":
                model.reset(rf.where("address") == "/", descendants=True)
            model.eval()
            current = model.state_dict()
            changes = [name for name, value in state.items() if name not in current or not equal(value, current[name])]
            before = prediction(model, test)
            immediate = scores(test, before, means)
            if arm == "selective_reset":
                checks["Reset changes selected learned state"] = any(name.startswith("nodes./u.") for name in changes)
                checks["Reset preserves every unselected state entry"] = all(
                    name.startswith("nodes./u.") for name in changes
                )
                checks["Reset preserves the schema"] = model.schema.model_dump() == source.schema.model_dump()
                checks["Selected target immediately loses its skill"] = (
                    immediate["u"]["nrmse"] > 0.8 and immediate["u"]["nrmse"] > 3 * initial["u"]["nrmse"]
                )
                checks["Unselected target immediately preserves predictions"] = bool(
                    np.allclose(reference["v"], before["v"], rtol=1e-5, atol=1e-6 * scales["v"])
                )
                head = {name: p.detach().clone() for name, p in model.nodes["/u"].named_parameters()}
            if arm == "complete_reset":
                checks["Complete reset loses both learned tasks"] = all(
                    immediate[name]["nrmse"] > 0.8 and immediate[name]["nrmse"] > 3 * initial[name]["nrmse"]
                    for name in TARGETS
                )
            fitting = fit(model, seed=seed + 200, split=seed + 11, budget=adapt_budget, accelerator=accelerator)
            after = prediction(model, test)
            final = scores(test, after, means)
            arms[arm] = {
                "before": immediate,
                "after": final,
                "changed_state_entries": changes,
                "initial_prediction_drift": {
                    name: float(np.max(np.abs(reference[name] - before[name]))) for name in TARGETS
                },
                **fitting,
            }
            checks[f"{arm}: fresh optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            checks[f"{arm}: both final targets below 0.25 nRMSE"] = all(
                value["nrmse"] < 0.25 for value in final.values()
            )
            if arm == "selective_reset":
                changed = [name for name, p in model.nodes["/u"].named_parameters() if not torch.equal(head[name], p)]
                arms[arm]["reset_head_updated_parameters"] = changed
                checks["Reset head parameters actually relearn"] = bool(changed)
                corrupted = prediction(model, test, corrupt=True)
                checks["Hidden target placeholders cannot affect predictions"] = all(
                    np.allclose(after[name], corrupted[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
                path = Path(directory) / "relearned.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                restored = prediction(loaded, test)
                checks["Relearned checkpoint preserves schema and state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                )
                checks["Relearned checkpoint preserves predictions"] = all(
                    np.allclose(after[name], restored[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
    return {
        "source": initial,
        "source_fit": source_fit,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "arms": arms,
        "test_rows": len(test),
        "baseline": "Source-training mean for test comparisons; phase-training mean for validation curves",
        "optimizer_policy": "New AdamW factory and Trainer for every fit; both hidden labels rehearsed",
        "mutation": "reset /u; complete-reset control resets record with descendants=True",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4801)

# %% [markdown]
# </details>
