/** Reader-facing explanations and comparison contracts. Measurements remain in CI publications. */
export type Metric =
  | 'RMSE'
  | 'MAE'
  | 'MSE'
  | 'Normalized RMSE'
  | 'Normalized MAE'
  | 'AUC'
  | 'Accuracy'
  | 'Probability'
  | 'Proportion'
  | 'Prediction drift'
  | 'Assigned groups'
  | 'Predicted value'
  | 'Adjusted Rand index'
  | 'Macro F1'
  | 'False positive rate'
  | 'Probability RMSE'
  | 'Checks met';
export type ComparisonKind =
  | 'Controlled comparison'
  | 'Baseline reference'
  | 'Before and after'
  | 'Recorded conditions'
  | 'Recorded measurement';
export interface Review {
  question: string;
  description?: string;
  modeling: string;
  note: string;
  comparison: {
    kind: ComparisonKind;
    metric: Metric;
    conditions: { label: string; path: (string | number)[]; format: string }[];
  };
}

export const reviews: Record<string, Review> = {
  P001: {
    question: 'Can relflow add a list of numbers?',
    modeling:
      'Generate independent lists of one to six amounts between −1 and 1. Give the model the amounts in a repeated branch and hide their sum as a Number target. Training, validation, and test use separate lists. Compare test RMSE with a predictor that always returns a constant.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: '',
  },
  P002: {
    question: 'Can relflow add longer lists than relflow saw in training?',
    modeling:
      'Train on lists containing one to six signed amounts, with their total hidden. Test on new lists containing seven to ten amounts. The model receives the individual amounts without a separate count field. Compare its error on familiar and longer lengths.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Familiar lengths',
          path: ['in_range_score', 'rmse'],
          format: 'error',
        },
        {
          label: 'Longer lists',
          path: ['unseen_score', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The branch can hold these lists; this tests learned generalization to lengths seven through ten, not arbitrary-length arithmetic.',
  },
  P003: {
    question: 'Does repeating a list preserve its average?',
    modeling:
      'Generate lists of numbers and hide their arithmetic average. A Mean branch averages their encoded values, with item attention disabled. Train a decoder to recover the numerical average, then score independent lists against a constant predictor. Reordering items and repeating the entire list should leave the answer unchanged.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Permutation and repetition stability are separate checks below; low prediction error alone does not establish either property.',
  },
  P004: {
    question: 'Can a visible count help recover a total?',
    modeling:
      'Repeat one randomly chosen amount between one and six times. Give the model both the repeated items and their count, and hide the total. A Mean branch supplies the item representation; the visible count supplies the multiplier. Compare held-out total error with a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'This is a supplied-count control. It does not show that the model inferred the count without an explicit input.',
  },
  P005: {
    question: 'Does the count help with longer lists?',
    modeling:
      'Train a model to predict a repeated amount’s total from the items and a visible count. Training counts range from one to six; test counts range from seven to ten. Duplicating a test list also doubles the supplied count and expected total.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['unseen_score', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['unseen_score', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The primary comparison measures larger-count prediction error. It does not isolate the effect of removing the count input.',
  },
  P006: {
    question: 'Can relflow learn how two measurements vary together?',
    modeling:
      'Generate eight paired x and y measurements with varying means, spreads, and correlation. Hide their covariance: the average product of their centered values. Compare correct pairs with a test copy that shifts y between items while retaining the original answers. Both copies contain exactly the same individual x and y values.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Broken pairs',
          path: ['corrupted', 'rmse'],
          format: 'error',
        },
        {
          label: 'Correct pairs',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The shuffled-pair condition tests whether the model uses pairing. The target is covariance, not correlation.',
  },
  P007: {
    question: 'Can relflow tell how spread out a list is?',
    modeling:
      'Generate lists of eight raw numbers with independently varying location and spread. Hide their variance, the average squared distance from the list mean. The model receives no prepared mean or squared differences. Compare held-out variance error with a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'A separate paired check holds the mean fixed while changing the spread, so location alone cannot explain success.',
  },
  P008: {
    question: 'Can relflow average prepared contributions to covariance?',
    modeling:
      'Calculate each item’s contribution to covariance upstream: (x − mean(x)) × (y − mean(y)). Give those eight prepared contributions to a Mean branch and hide their average. Compare the learned prediction with a constant predictor on fresh records.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Centering and multiplication are already supplied. This diagnostic tests averaging and decoding, not learning covariance from raw pairs.',
  },
  P009: {
    question: 'Can relflow average prepared squared differences?',
    modeling:
      'Calculate each value’s squared distance from its list mean before giving data to the model. A Mean branch receives the eight prepared squared differences; the hidden target is their average, or population variance. Compare held-out error with a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The model does not learn centering or squaring here. P007 tests variance from raw values.',
  },
  P010: {
    question: 'Can relflow total weighted values for one group?',
    modeling:
      'Give each item a group, value, and weight. A visible request selects one of three groups; the hidden answer is that group’s sum of value × weight. Compare normal test records with copies that swap weights within groups while preserving the original answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Swapped weights',
          path: ['pairing', 'rmse'],
          format: 'error',
        },
        {
          label: 'Correct data',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Changing group labels is a second control. Neither control supplies precomputed products to the model.',
  },
  P011: {
    question: 'Can relflow total prepared contributions for one group?',
    modeling:
      'Give each item a group and an already calculated value × weight contribution. Ask for one group’s total and hide that answer. Compare normal records with copies whose group labels are rotated while contributions and expected totals stay fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Changed groups',
          path: ['corrupted', 'rmse'],
          format: 'error',
        },
        {
          label: 'Correct groups',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Products are supplied upstream. This isolates group selection and summation from learning multiplication.',
  },
  P012: {
    question: 'Can relflow find a requested value in an unsorted list?',
    modeling:
      'Give the model an unsorted list of nine numbers and a request for the minimum, lower quarter, median, upper quarter, or maximum. Hide the requested value. Compare correct requests with changed requests scored against the original answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Changed requests',
          path: ['cycled', 'rmse'],
          format: 'error',
        },
        {
          label: 'All requests',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'All five ranks are checked separately. With nine values, the quarter positions are observed items, so interpolated quantiles are not tested.',
  },
  P013: {
    question: 'Can relflow predict the total of repeated values?',
    modeling:
      'Repeat the same contribution one to six times. Ask a Mean branch, with item attention disabled, to predict both the average and total. Its averaged content is identical across repetition counts, but its additional count signal can distinguish them. The primary comparison measures total error against a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['sum_score', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['sum_score', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'This implementation’s Mean summary carries count information. Passing does not show that a plain arithmetic average alone retains multiplicity.',
    description:
      'Predict a total from a Mean summary that retains both averaged content and item count.',
  },
  P014: {
    question: 'Does each value keep its own weight?',
    modeling:
      'Generate six items, each with its own independently sampled value and positive weight. Hide the sum of their value × weight products. Compare correctly paired items with a copy that swaps weights between items while keeping the original total.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Swapped weights',
          path: ['corrupted', 'rmse'],
          format: 'error',
        },
        {
          label: 'Correct pairs',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Reordering whole items and scaling every weight are additional checks. Swapping weights tests a different property: retaining each value’s own partner.',
  },
  P015: {
    question: 'Can relflow total contributions already calculated?',
    modeling:
      'Compute value × weight for each of six items before loading the record. Give only these contributions to the model and hide their sum. Compare held-out total error with a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['measured', 'baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['measured', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'This diagnostic supplies the multiplication step. P014 tests whether the model can learn it from separate values and weights.',
  },
  P016: {
    question: 'Can relflow learn a weighted average?',
    modeling:
      'Generate lists of two to six value–weight pairs. Hide the weighted average: sum(value × weight) divided by sum(weight). Compare normal test records with copies that swap weights between values while preserving the original answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Swapped weights',
          path: ['corrupted', 'rmse'],
          format: 'error',
        },
        {
          label: 'Correct pairs',
          path: ['intact', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Repeating whole items or scaling all weights should preserve the answer; these invariance checks remain separate from prediction error.',
  },
  P017: {
    question: 'Can relflow learn a signal without inventing one?',
    modeling:
      'Train two yes/no predictors with the same architecture and budget. In one dataset, the visible Boolean matches the hidden answer. In the control, the visible Boolean and balanced answer are independent. Evaluate both on fresh records using AUC.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'AUC',
      conditions: [
        {
          label: 'Unrelated answers',
          path: ['noise_auc'],
          format: 'auc',
        },
        {
          label: 'Matching answers',
          path: ['signal_auc'],
          format: 'auc',
        },
      ],
    },
    note: 'The unrelated-input control should remain near chance. High accuracy on its training rows would not establish a learnable test signal.',
  },
  P018: {
    question: 'Does the archived Cluster field recover five behavioral groups?',
    modeling:
      'Generate repeated observations of 100 identities, each following one of five hidden numerical rules. The archived Cluster field reconstructs identities while a numeric target encourages learning their behavior. Inspect the final number of assigned groups.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Assigned groups',
      conditions: [
        {
          label: 'Groups at the final epoch',
          path: ['final_committed', -1],
          format: 'number',
        },
      ],
    },
    note: 'Group count and usage are diagnostics, not a test of whether the correct identities were grouped together. This historical run also reuses training data for validation.',
  },
  P019: {
    question: 'Does an input-only Cluster field start grouping by itself?',
    modeling:
      'Give the archived Cluster field recurring identities as inputs and train a sibling label predictor. Do not add a Cluster reconstruction objective. Track whether the field’s adaptive group count and activity change during fitting.',
    comparison: {
      metric: 'Assigned groups',
      kind: 'Before and after',
      conditions: [
        {
          label: 'First recorded epoch',
          path: ['committed', 0],
          format: 'number',
        },
        {
          label: 'Final recorded epoch',
          path: ['committed', -1],
          format: 'number',
        },
      ],
    },
    note: 'The checks intentionally require the grouping mechanism to remain inactive. Checks met does not mean automatic grouping succeeded.',
  },
  P020: {
    question: 'Does reconstructing identities produce the expected number of Cluster groups?',
    modeling:
      'Generate recurring identities with five stable hidden labels. Train the archived Cluster field to reconstruct identities while predicting the labels. Inspect how many groups are assigned after training.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Assigned groups',
      conditions: [
        {
          label: 'Groups at the final epoch',
          path: ['final_committed', -1],
          format: 'number',
        },
      ],
    },
    note: 'A group count near five does not establish correct membership recovery. The recorded checks concern group count and usage, not an independently scored partition.',
  },
  P021: {
    question: 'Can relflow recognize matching IDs relflow has never seen?',
    modeling:
      'Give the model two identifiers and hide a Boolean target indicating whether they match. Training and test use disjoint identifier names. Compare Hash with Category on the same equality task: new Category labels share the unknown representation, whereas Hash retains equality within an encoded batch. A shuffled-label control checks for an accidental signal.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'AUC',
      conditions: [
        {
          label: 'Unknown Category inputs',
          path: ['category_oov_auc'],
          format: 'auc',
        },
        {
          label: 'Hash inputs',
          path: ['hash_auc'],
          format: 'auc',
        },
      ],
    },
    note: '',
  },
  P022: {
    question: 'Can a summary retain the winning item’s value?',
    modeling:
      'Give each candidate a score and a separate payload value. Hide the payload belonging to the highest score. Compress the candidate collection into learned summaries, then compare correct records with copies that rotate payloads between candidates while retaining the original answer.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Broken pairs',
          path: ['broken_nrmse'],
          format: 'error',
        },
        {
          label: 'Correct pairs',
          path: ['intact_nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'The control preserves the separate score and payload distributions. It breaks their association, which is the information needed to retrieve the winner.',
  },
  P023: {
    question: 'Can relflow return the winning item’s value?',
    modeling:
      'Give each candidate a score and payload, retain the encoded candidates, and hide the payload with the highest score. Compare correct records with copies that rotate payloads while leaving scores and expected answers fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Broken pairs',
          path: ['broken_nrmse'],
          format: 'error',
        },
        {
          label: 'Correct pairs',
          path: ['intact_nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'P022 asks the same retrieval question after compression. This page tests the route that retains candidates.',
  },
  P024: {
    question: 'Can matching positions make an apparent ID lookup misleading?',
    modeling:
      'Place source values and their queries in corresponding positions, with matching IDs. Hide the copied values. Then rotate query IDs without moving the queries or changing their expected answers. Compare error before and after the ID change.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Changed IDs',
          path: ['broken_nrmse'],
          format: 'error',
        },
        {
          label: 'Matching IDs',
          path: ['intact_nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'The checks intentionally test a positional shortcut. Remaining accurate after IDs change is not evidence of an ID-based lookup.',
  },
  P025: {
    question: 'Can relflow look up a value using a new ID?',
    modeling:
      'Put IDs and values in a source list, and ask for those values in an independently shuffled query list. Use unfamiliar IDs at evaluation so position and memorized names cannot supply the answer. Compare matching IDs with rotated query IDs, retaining the original targets.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Changed IDs, median of three seeds',
          path: ['median', 'broken_nrmse'],
          format: 'error',
        },
        {
          label: 'Matching IDs, median of three seeds',
          path: ['median', 'intact_nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'The chart shows median errors across the three recorded seeds. Every seed and the worst-seed result remain available below.',
  },
  P026: {
    question: 'Can relflow average only the requested group?',
    modeling:
      'Mix values from three named groups in one list. Supply a group request and hide that group’s average. Compare normal records with copies that rotate item group labels without changing values or expected answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Group A, changed labels',
          path: ['corrupted/A/rmse'],
          format: 'error',
        },
        {
          label: 'Group A, correct labels',
          path: ['intact/A/rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The chart highlights group A. The full checks require the relationship separately for groups A, B, and C.',
  },
  P027: {
    question: 'Can relflow answer all twelve group-and-operation requests?',
    modeling:
      'Give each item a group and value, then request one group’s minimum, maximum, average, or total. Hide the requested answer. Generate independent lists for training and testing, and measure every combination of three groups and four operations.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Min · group A',
          path: ['intact/min/A/nrmse'],
          format: 'number',
        },
        {
          label: 'Min · group B',
          path: ['intact/min/B/nrmse'],
          format: 'number',
        },
        {
          label: 'Min · group C',
          path: ['intact/min/C/nrmse'],
          format: 'number',
        },
        {
          label: 'Max · group A',
          path: ['intact/max/A/nrmse'],
          format: 'number',
        },
        {
          label: 'Max · group B',
          path: ['intact/max/B/nrmse'],
          format: 'number',
        },
        {
          label: 'Max · group C',
          path: ['intact/max/C/nrmse'],
          format: 'number',
        },
        {
          label: 'Mean · group A',
          path: ['intact/mean/A/nrmse'],
          format: 'number',
        },
        {
          label: 'Mean · group B',
          path: ['intact/mean/B/nrmse'],
          format: 'number',
        },
        {
          label: 'Mean · group C',
          path: ['intact/mean/C/nrmse'],
          format: 'number',
        },
        {
          label: 'Sum · group A',
          path: ['intact/sum/A/nrmse'],
          format: 'number',
        },
        {
          label: 'Sum · group B',
          path: ['intact/sum/B/nrmse'],
          format: 'number',
        },
        {
          label: 'Sum · group C',
          path: ['intact/sum/C/nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The chart compares errors across the twelve tasks, not an ablation. All twelve must satisfy their own criteria.',
  },
  P028: {
    question: 'Can relflow find a shared ID across two lists?',
    modeling:
      'Generate two lists of previously unseen IDs. Half the records share exactly one ID; half share none. Hide the overlap answer, then remove the shared member from positive test records while retaining their original labels. Compare ranking scores before and after that change.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'AUC',
      conditions: [
        {
          label: 'Match removed',
          path: ['broken_auc'],
          format: 'auc',
        },
        {
          label: 'Original lists',
          path: ['intact_auc'],
          format: 'auc',
        },
      ],
    },
    note: 'This experiment’s checks deliberately characterize near-chance overlap detection. Checks met means that measured limitation was reproduced, not that overlap detection succeeded.',
  },
  P029: {
    question: 'Can relflow compare two unfamiliar IDs directly?',
    modeling:
      'Give two previously unseen IDs and hide whether they match. Use equal numbers of matching and nonmatching pairs. After fitting, replace one ID in every matching pair while retaining its original label. Compare AUC on intact and changed pairs.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'AUC',
      conditions: [
        {
          label: 'Matches removed',
          path: ['broken_auc'],
          format: 'auc',
        },
        {
          label: 'Original pairs',
          path: ['intact_auc'],
          format: 'auc',
        },
      ],
    },
    note: 'This isolates single-pair equality from the two-list overlap task in P028.',
  },
  P030: {
    question: 'Does each subtotal stay with its own item?',
    modeling:
      'Generate orders with zero to five items. Give each real item its own quantity and price, and hide its subtotal. Compare predictions with the correct item-level answers and with answers shuffled between items while keeping all visible inputs fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Shuffled item answers',
          path: ['permuted_target_nrmse'],
          format: 'error',
        },
        {
          label: 'Correct item answers',
          path: ['aligned_nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'The changed-target control tests output alignment. Separate checks preserve empty orders, exact collection lengths, and inferred flags.',
  },
  P031: {
    question: 'Can an instruction change the calculation?',
    modeling:
      'Give the same list a visible request for minimum, maximum, average, or total. Hide the requested result. Compare total-prediction error with the request visible and hidden; without the request, different questions about one list become indistinguishable.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Total, instruction hidden',
          path: ['hidden/sum/rmse'],
          format: 'error',
        },
        {
          label: 'Total, instruction visible',
          path: ['intact/sum/rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The chart shows the total task. Minimum, maximum, average, and item-order stability each have separate checks.',
  },
  P032: {
    question: 'Can relflow compare an item with its own group?',
    modeling:
      'Give each item a value and group. Hide its difference from that group’s average, producing one answer per item. Compare normal records with copies whose group labels are rotated while values and original answers stay fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Changed groups',
          path: ['corrupted_rmse'],
          format: 'error',
        },
        {
          label: 'Correct groups',
          path: ['intact_rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Additional tests shift every value in a group equally and reorder whole items; both should preserve the corresponding deviations.',
  },
  P033: {
    question: 'Can one summary support comparisons within groups?',
    modeling:
      'Give each item a group and value, compress the collection into one learned summary, and hide each item’s difference from its group mean. Compare correct labels with rotated labels while retaining all values and original answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Changed groups',
          path: ['corrupted_rmse'],
          format: 'error',
        },
        {
          label: 'Correct groups',
          path: ['intact_rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Each output can still see its own item’s visible fields. The test asks whether the shared summary retains the peer context it also needs.',
  },
  P034: {
    question: 'Can relflow subtract an average supplied as input?',
    modeling:
      'Give every item both its value and an already calculated peer average. Hide their difference. Vary the overall location between records so the raw value alone is not a useful shortcut, and compare held-out subtraction error with a constant predictor.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['intact_rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The average is supplied upstream. This isolates local subtraction and repeated-output alignment from learning an aggregate.',
  },
  P035: {
    question: 'Can relflow infer an average and compare each item?',
    modeling:
      'Give a list of six values and hide each item’s difference from the list average. Do not provide the average. Compare error on normal lists and on copies with the same constant added to every value, which leaves all correct deviations unchanged.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Original lists',
          path: ['intact_rmse'],
          format: 'error',
        },
        {
          label: 'Shifted lists',
          path: ['translated_rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Reordering items is checked separately: the predictions should move with their items.',
  },
  P036: {
    question: 'Can familiar IDs connect two lists?',
    modeling:
      'Give a source list of category IDs and changing values, then a separately ordered query list using familiar IDs. Hide each requested source value. Compare correct query IDs with rotated IDs that retain their original targets.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Query IDs changed',
          path: ['preserved_broken_nrmse'],
          format: 'number',
        },
        {
          label: 'Matching familiar IDs',
          path: ['preserved_nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'This comparison uses the route that retains source items. Compressed-route results remain below; it does not test unseen Category labels.',
  },
  P037: {
    question: 'Can new IDs connect two lists?',
    modeling:
      'Give source IDs and values, and a separately shuffled list of queries. Use Hash IDs absent from training and hide each requested value. Compare correct IDs with rotated query IDs while retaining the same source records and expected answers.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Query IDs changed',
          path: ['preserved_broken_nrmse'],
          format: 'number',
        },
        {
          label: 'Matching unseen Hash IDs',
          path: ['preserved_nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The primary comparison retains source items. The additional compression comparison measures what information is lost when those items are summarized.',
  },
  P038: {
    question: 'Can relflow distinguish missing values from zero?',
    modeling:
      'Generate records in which a measurement is either null or the number zero, and hide the Boolean answer identifying which case occurred. Compare preserving these states with replacing nulls by zero. Filling nulls removes the information needed to distinguish the two cases; the test measures whether the model uses the preserved distinction.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Nulls filled with zero',
          path: ['prefilled_accuracy'],
          format: 'percent',
        },
        {
          label: 'Nulls preserved',
          path: ['state_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: '',
  },
  P039: {
    question: 'Can relflow add six numbers?',
    modeling:
      'Generate independent records containing exactly six amounts sampled between −1 and 1. Put the amounts in a repeated branch and hide their sum as a Number target. Score held-out totals with RMSE and compare with a predictor that always returns the training-set average total.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['baseline_rmse'],
          format: 'error',
        },
        {
          label: 'Trained model',
          path: ['test_rmse'],
          format: 'error',
        },
      ],
    },
    note: '',
  },
  P040: {
    question: 'Does the grouping of values matter?',
    modeling:
      'Pair records containing the same six amounts, arranged in sessions of two, two, and two versus one, four, and one. Hide the largest session total. Compare the nested model with the best possible squared-error prediction available after flattening away session membership.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Best flattened-data reference',
          path: ['flat_oracle_rmse'],
          format: 'error',
        },
        {
          label: 'Model with nested sessions',
          path: ['test_rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The flattened-data reference is an information limit, not a competing trained model. This tests sensitivity to these session partitions.',
  },
  P041: {
    question: 'Can relflow distinguish lists with the same average but different totals?',
    modeling:
      'Repeat the same amount one to six times and hide both the average and total. Disable item attention and summarize the collection with Mean, which also carries a count signal. Compare the learned total with a constant predictor on held-out records.',
    comparison: {
      metric: 'RMSE',
      kind: 'Baseline reference',
      conditions: [
        {
          label: 'Constant-prediction baseline',
          path: ['total_baseline_rmse'],
          format: 'number',
        },
        {
          label: 'Predicted total',
          path: ['total_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'A separate pair uses one versus six copies of the same amount. This implementation tests the count-augmented Mean summary, not a plain average that discards count.',
    description:
      'Learn separate averages and totals from repeated values summarized with a count signal.',
  },
  P042: {
    question: 'Does relflow need both weekday and hour?',
    modeling:
      'Generate weekday business-hour observations and two kinds of near miss: weekdays outside the window, and weekends inside it. Hide whether the business is open. Train otherwise matched models with weekday only, hour only, or both, and compare accuracy on fresh dates.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Weekday only',
          path: ['day_only_accuracy'],
          format: 'percent',
        },
        {
          label: 'Hour only',
          path: ['hour_only_accuracy'],
          format: 'percent',
        },
        {
          label: 'Weekday and hour',
          path: ['composed_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'Each single coordinate leaves ambiguous cases. The comparison tests combining the two visible calendar parts.',
  },
  P043: {
    question: 'Can day 60 reveal the month in a leap year?',
    modeling:
      'Choose dates where day 60 is either February 29 or March 1. With only day-of-year visible, the inputs are identical but the answers differ. Compare that model with one also given week-of-month, which resolves the ambiguity.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Day number only',
          path: ['day_only_accuracy'],
          format: 'percent',
        },
        {
          label: 'Week-of-month added',
          path: ['with_week_of_month_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The two runs use different training budgets. This is an information-availability comparison, not a matched compute ablation.',
  },
  P044: {
    question: 'Can relflow learn the month from the day number?',
    modeling:
      'Give a date’s day-of-year and hide its month, using non-leap years only. Train on odd-numbered days and test on even-numbered days in unseen years. Compare the original test dates with timestamps shuffled between records while their month answers stay fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Shuffled dates',
          path: ['permuted_accuracy'],
          format: 'percent',
        },
        {
          label: 'Original dates',
          path: ['periodic_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'This tests interpolation of non-leap month boundaries. Leap-year ambiguity is tested separately in P043.',
  },
  P045: {
    question: 'Can day-of-year identify weekday without the missing calendar information?',
    modeling:
      'Balance all seven weekdays for every sampled day-of-year, so the visible day number alone cannot identify the answer. Compare a model given only day-of-year with one also given the weekday coordinate itself.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Day number only',
          path: ['day_only_accuracy'],
          format: 'percent',
        },
        {
          label: 'Weekday supplied',
          path: ['with_weekday_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The control supplies the answer-bearing calendar part directly. It does not test calculating weekday from a year.',
  },
  P046: {
    question: 'Do harmless schema edits preserve predictions?',
    modeling:
      'Train a predictor, then change metadata or add and remove inactive fields without changing its useful inputs. Compare predictions on the same test records before and after each edit. The primary chart shows the largest prediction change for each edit in the first cycle.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Prediction drift',
      conditions: [
        {
          label: 'Metadata edit',
          path: ['edits', 0, 'max_prediction_drift'],
          format: 'number',
        },
        {
          label: 'Inactive extension',
          path: ['edits', 1, 'max_prediction_drift'],
          format: 'number',
        },
        {
          label: 'Inactive deletion',
          path: ['edits', 2, 'max_prediction_drift'],
          format: 'number',
        },
      ],
    },
    note: 'Zero drift means identical predictions for that edit. Repeated cycles, equivalent key changes, checkpoint round-trips, and rejected edits are checked separately.',
  },
  P047: {
    question: 'What happens when a new prediction target is added?',
    modeling:
      'Train a model to predict two hidden numeric targets from inputs a and b. Add a third hidden target, w, without replacing the existing heads. Measure w’s RMSE immediately after the edit and after continued training on all targets.',
    comparison: {
      kind: 'Before and after',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Before continued training',
          path: ['arms', 'extended', 'before', 'w', 'rmse'],
          format: 'error',
        },
        {
          label: 'After continued training',
          path: ['arms', 'extended', 'after', 'w', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Learning the new target is only part of the test. Separate criteria check preservation of existing state and predictions, optimizer coverage, and save/load.',
  },
  P048: {
    question: 'Can one prediction target be reset and relearned?',
    modeling:
      'Train two hidden numeric targets, then reset only the learned head for u. Continue training both targets on the same task. Compare u’s RMSE immediately after reset with its error after retraining.',
    comparison: {
      kind: 'Before and after',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Before continued training',
          path: ['arms', 'selective_reset', 'before', 'u', 'rmse'],
          format: 'error',
        },
        {
          label: 'After continued training',
          path: ['arms', 'selective_reset', 'after', 'u', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The sibling v must retain its state and immediate predictions. A complete-reset control checks that erasing both heads actually removes both learned skills.',
  },
  P049: {
    question: 'Can an input be disabled and restored?',
    modeling:
      'Train on y = a + b, then temporarily disable input b without retraining. Score the original model, the disabled-input model, and the model after b is restored on the same test rows.',
    comparison: {
      kind: 'Before and after',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Original model',
          path: ['source', 'rmse'],
          format: 'error',
        },
        {
          label: 'Input disabled · first cycle',
          path: ['edits', 0, 'scores', 'rmse'],
          format: 'error',
        },
        {
          label: 'Input restored · first cycle',
          path: ['edits', 1, 'scores', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Restoring an inactive input can reuse its learned state. Separate checks cover repeated cycles, temporary overrides, and checkpoints; deletion is tested in P050.',
  },
  P050: {
    question: 'Can relflow adapt after losing an input?',
    modeling:
      'Train on y = a + b, permanently delete b, and continue fitting with only a. Compare the adapted model’s RMSE with the best prediction available from a alone. Independent b values now contribute information that no remaining input can recover.',
    comparison: {
      metric: 'RMSE',
      kind: 'Baseline reference',
      conditions: [
        {
          label: 'Best possible prediction from a alone',
          path: ['only_a_oracle', 'rmse'],
          format: 'number',
        },
        {
          label: 'Model after deleting b and adapting',
          path: ['arms', 'deleted', 'after', 'rmse'],
          format: 'number',
        },
      ],
    },
    note: 'The reference is an information limit, not a trained competitor. Separate controls compare deletion, deactivation, and a fresh restricted model.',
  },
  P051: {
    question: 'Can hiding values teach a relationship in both directions?',
    modeling:
      'Generate independent x values and set y = 2x + 0.3. Randomly hide either field during training so the model learns to reconstruct it from the other. At test time, request each direction and compare intact context with shuffled context.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Recover x · shuffled y',
          path: ['directions', 'x', 'shuffled_nrmse'],
          format: 'number',
        },
        {
          label: 'Recover x · correct y',
          path: ['directions', 'x', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Recover y · shuffled x',
          path: ['directions', 'y', 'shuffled_nrmse'],
          format: 'number',
        },
        {
          label: 'Recover y · correct x',
          path: ['directions', 'y', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The four bars use normalized error so both directions can be shown on comparable reference scales. Hidden-value poisoning is checked separately.',
  },
  P052: {
    question: 'Can each item choose which value to hide?',
    modeling:
      'Generate bags of items with x and value = 2x + 0.3. An item-level Boolean selector hides some values and requests their reconstruction. Score only selected real items, comparing their original x values with independently shuffled context.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Normalized RMSE',
      conditions: [
        {
          label: 'Shuffled visible x',
          path: ['corrupted_context_nrmse'],
          format: 'error',
        },
        {
          label: 'Correct visible x',
          path: ['nrmse'],
          format: 'error',
        },
      ],
    },
    note: 'Selectors must also match training objectives and output flags exactly, including unselected and padded positions.',
  },
  P053: {
    question: 'Can relflow recover when a whole branch is hidden?',
    modeling:
      'Create two branches carrying redundant measurements of one hidden signal. Randomly hide whole branches during fitting. On the same test records, measure error with both views, either view alone, and neither view available.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Neither view available',
          path: ['patterns', 'hiddenhidden', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Only first view available',
          path: ['patterns', 'visiblehidden', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Only second view available',
          path: ['patterns', 'hiddenvisible', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Both views available',
          path: ['patterns', 'visiblevisible', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The target is identical in every condition. The no-view condition checks the information boundary; hidden descendants must never leak into the answer.',
  },
  P054: {
    question: 'Can relflow predict the next value without seeing the future?',
    modeling:
      'Generate straight-line sequences with a fresh starting value and increment for every record. Show only the first two to four values and hide the future. For the primary comparison, roll predictions forward from two true starting values and compare with always repeating the last observed value.',
    comparison: {
      metric: 'RMSE',
      kind: 'Baseline reference',
      conditions: [
        {
          label: 'Repeat the last observed value',
          path: ['rollout', 'persistence_rmse'],
          format: 'number',
        },
        {
          label: 'Model rollout',
          path: ['rollout', 'rmse'],
          format: 'number',
        },
      ],
    },
    note: 'This is a short synthetic linear rollout, not a general time-series forecast. Separate tests score each prefix length and verify that hidden future values cannot leak.',
  },
  P055: {
    question: 'Is an unfamiliar category the same as a null?',
    modeling:
      'Generate numeric context x and a category-specific effect, then hide 2x + effect. At test time, introduce two unseen categories with opposite effects. Both map to unknown content, so the model can use x but cannot distinguish those effects. Compare its error with that unavoidable information limit.',
    comparison: {
      kind: 'Baseline reference',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Unavoidable error without identity',
          path: ['unknown', 'information_limit_rmse'],
          format: 'number',
        },
        {
          label: 'Error on paired new identities',
          path: ['unknown', 'paired_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'Unknown category content remains different from a null. The reference is a mathematical information limit, not another trained model.',
  },
  P056: {
    question: 'Can perfect accuracy hide answers relflow cannot name?',
    modeling:
      'Train a Category output to name two classes from numeric evidence. In half the test rows, replace the expected name with an unseen target label while keeping the evidence unchanged. Compare accuracy on known answers with accuracy across every test row.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Known-label accuracy',
          path: ['known_accuracy'],
          format: 'percent',
        },
        {
          label: 'All-row accuracy',
          path: ['all_row_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'These scores have different denominators. Perfect known-answer accuracy can coexist with zero ability to emit the missing target names.',
  },
  P057: {
    question: 'Does label encounter order limit vocabulary coverage?',
    modeling:
      'Present the same four observed labels in two orders: rare labels first or common labels first. Let the input vocabulary grow automatically while learning each label’s numerical effect. Compare how many common labels are represented after training.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Common-label coverage, rare first',
          path: ['arms', 'rare_first', 'common_coverage'],
          format: 'percent',
        },
        {
          label: 'Common-label coverage, common first',
          path: ['arms', 'common_first', 'common_coverage'],
          format: 'percent',
        },
      ],
    },
    note: 'Coverage measures admitted labels, not predictive accuracy. Additional gates require all four labels to be stored and their effects to be learned.',
  },
  P058: {
    question: 'What does an untrained Set fallback do with entirely new members?',
    modeling:
      'Train on known set members, empty sets, and nulls, without simulating unavailable members. Predict a numeric function of membership. Then compare an empty input with an input containing only unseen member names.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Predicted value',
      conditions: [
        {
          label: 'Prediction for empty input',
          path: ['boundary_predictions', 'empty'],
          format: 'number',
        },
        {
          label: 'Prediction for all-unknown input',
          path: ['boundary_predictions', 'all_unknown'],
          format: 'number',
        },
      ],
    },
    note: 'The checks deliberately expose the untrained fallback: empty and all-unknown inputs can receive the same prediction, although null is distinct. P064 trains that fallback explicitly.',
  },
  P059: {
    question: 'Does admitting a label teach relflow to predict that label?',
    modeling:
      'Train a Category output on two numeric regions and their labels. Admit labels for two new regions without an optimizer update, then continue fitting with both old and new examples. Compare new-class accuracy immediately after admission and after learning.',
    comparison: {
      kind: 'Before and after',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'New labels, just admitted',
          path: ['new_accuracy_immediately_after_admission'],
          format: 'percent',
        },
        {
          label: 'New labels, after fitting',
          path: ['final_new_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'Adding a name to the vocabulary is not learning its meaning. Retention of old classes and evaluation-time vocabulary freezing are separate checks.',
  },
  P060: {
    question: 'Can relflow learn a 90% probability without a useful input?',
    modeling:
      'Generate yes/no answers independently of the input, with yes occurring 90% of the time. Train the model to predict the answer distribution. Compare its average predicted yes probability on familiar and unseen input IDs with the generator’s recorded probability.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Probability',
      conditions: [
        {
          label: 'True yes probability',
          path: ['prior'],
          format: 'percent',
        },
        {
          label: 'Predicted, known inputs',
          path: ['capacity_2_known', 'mean_probability'],
          format: 'percent',
        },
        {
          label: 'Predicted, new inputs',
          path: ['capacity_2_novel', 'mean_probability'],
          format: 'percent',
        },
      ],
    },
    note: 'This chart shows probabilities, not accuracy. The full criteria also measure per-record probability error and expected log loss.',
  },
  P061: {
    question: 'How confident should relflow be about a new identity?',
    modeling:
      'Generate outcomes from a numeric signal plus an identity-specific effect. New identities remove only the second source of information. Train with some input identities made unavailable, then compare confidence on unseen IDs when the remaining numeric evidence is weak or decisive.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Probability',
      conditions: [
        {
          label: 'Weak signal, new identity, 20% training rate',
          path: ['0.2', 'novel', 'ambiguous_confidence'],
          format: 'percent',
        },
        {
          label: 'Strong signal, new identity, 20% training rate',
          path: ['0.2', 'novel', 'decisive_confidence'],
          format: 'percent',
        },
      ],
    },
    note: 'These conditions should have different confidence levels; higher confidence is not always better. Both bars use the 20% input-unavailability training condition.',
  },
  P062: {
    question: 'Does reconstruction accuracy include unfamiliar answers?',
    modeling:
      'Learn to reconstruct a hidden Category answer from numeric context. At evaluation, replace some expected target names with unseen labels while keeping the exact same inputs and selected positions. Compare accuracy among representable answers with accuracy across every selected answer.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Known-answer accuracy, low coverage',
          path: ['known_102', 'accuracy.content'],
          format: 'percent',
        },
        {
          label: 'All-answer accuracy',
          path: ['known_102', 'accuracy.all'],
          format: 'percent',
        },
      ],
    },
    note: 'The chart uses the approximately 10%-coverage condition. Known-answer scores exclude unavailable labels; coverage and all-answer accuracy expose that exclusion.',
  },
  P063: {
    question: 'Can relflow learn separate probabilities for Set members?',
    modeling:
      'Generate answer sets where red appears 90% of the time and blue 20%, independently of useful context. Hide the membership answer and train a Set decoder. Compare its predicted membership probability for each member with the stated generating frequencies.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Probability',
      conditions: [
        {
          label: 'Predicted red membership',
          path: ['capacity_2', 'probabilities', 'red'],
          format: 'percent',
        },
        {
          label: 'Predicted blue membership',
          path: ['capacity_2', 'probabilities', 'blue'],
          format: 'percent',
        },
      ],
    },
    note: 'Memberships are independent probabilities and need not add to one. Separate checks cover unknown members, coverage, and loss aggregation.',
  },
  P064: {
    question: 'Can relflow distinguish an unknown set from an empty set?',
    modeling:
      'Predict whether a set is nonempty. Train one model with input members sometimes made unavailable and another without that augmentation. Evaluate both on empty sets and sets containing only names absent from training, using the same hidden answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Unknown-input error, fallback untrained',
          path: ['control_unknown'],
          format: 'number',
        },
        {
          label: 'Unknown-input error, fallback trained',
          path: ['augmented_unknown'],
          format: 'number',
        },
      ],
    },
    note: 'The primary comparison is RMSE on unseen-member inputs. The task is nonemptiness, not recovering the identities of unseen members.',
  },
  P065: {
    question: 'Can a Cluster reconstruction score hide answers outside its vocabulary?',
    modeling:
      'Use the archived Cluster head to reconstruct hidden identities from a categorical cue. At evaluation, rename most selected targets to unseen identities without changing the cue or selection masks. Compare known-answer accuracy with accuracy across all selected targets.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Known-answer accuracy, low coverage',
          path: ['known_102', 'accuracy.content'],
          format: 'percent',
        },
        {
          label: 'All-answer accuracy',
          path: ['known_102', 'accuracy.all'],
          format: 'percent',
        },
      ],
    },
    note: 'The low-coverage condition tests metric interpretation. High known-answer accuracy does not establish broad identity coverage or recovery of meaningful groups.',
  },
  P066: {
    question: 'Can admitting a new input change predictions before learning?',
    modeling:
      'Train on one cohort of identities, then admit a new cohort from training data without an optimizer update. Predict the same held-out rows immediately before and after admission. Measure the largest change in predicted probability for each later cohort.',
    comparison: {
      metric: 'Prediction drift',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Second cohort · largest change',
          path: ['1', 'admission_max_drift'],
          format: 'number',
        },
        {
          label: 'Third cohort · largest change',
          path: ['2', 'admission_max_drift'],
          format: 'number',
        },
      ],
    },
    note: 'Unchanged predictions test the admission invariant only. Separate learning-quality gates must also hold after continued fitting.',
  },
  P067: {
    question: 'Can percentile-based predictions return useful original values?',
    modeling:
      'Generate skewed amounts and noisy costs following cost = 200 + 3 × amount. Represent both fields with Quantile and hide cost. Compare source-unit RMSE on fresh records with amounts shuffled between records while original costs stay fixed.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Shuffled amounts',
          path: ['shuffled_rmse'],
          format: 'error',
        },
        {
          label: 'Original amounts',
          path: ['rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The predictions are scored after conversion back to original units. Evaluation must not update either learned distribution.',
  },
  P068: {
    question: 'What happens beyond the learned percentile range?',
    modeling:
      'Train Quantile inputs and targets on amounts between 10 and 30, with cost = 100 + 4 × amount plus small noise. Test within that interval and in matched intervals below and above it. Compare RMSE in the original cost units.',
    comparison: {
      kind: 'Recorded conditions',
      metric: 'RMSE',
      conditions: [
        {
          label: 'In-range error, original units',
          path: ['in_range_rmse'],
          format: 'number',
        },
        {
          label: 'Below-range error, original units',
          path: ['below', 'rmse'],
          format: 'number',
        },
        {
          label: 'Above-range error, original units',
          path: ['above', 'rmse'],
          format: 'number',
        },
      ],
    },
    note: 'The checks deliberately measure tail saturation and bounded target support. Checks met does not mean the model extrapolated the relationship.',
  },
  P069: {
    question: 'Can relflow learn classes declared in advance?',
    modeling:
      'Declare four allowed input and output labels, then generate balanced route–channel combinations that determine a hidden class. Compare informative training with a matched model whose visible contexts are shuffled independently of the answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Shuffled training and test',
          path: ['shuffled_control', 'shuffled_test_accuracy'],
          format: 'percent',
        },
        {
          label: 'Informative training',
          path: ['informative', 'accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'All combinations occur during training. This tests learning a declared finite relationship, not zero-shot meaning or unseen-combination generalization.',
  },
  P070: {
    question: 'Does declaring a class teach its meaning?',
    modeling:
      'Declare four output classes, but initially train on examples from only three. Each class occupies a distinct noisy numeric region. Compare accuracy on the withheld class before and after continued fitting that includes its examples and rehearses the old classes.',
    comparison: {
      kind: 'Before and after',
      metric: 'Accuracy',
      conditions: [
        {
          label: 'Withheld class, before examples',
          path: ['initial_withheld', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'Withheld class, after examples',
          path: ['final_withheld', 'accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'A declared class can be named before it is learned. Separate checks retain old-class accuracy and freeze class IDs and exposure counts during evaluation.',
  },
  P071: {
    question: 'Can familiar instructions work in new combinations?',
    modeling:
      'Generate lists containing three groups and ask for a group’s mean, minimum, or maximum. Leave three group–operation combinations out of training. Compare error on the withheld coral-maximum request with a model trained on all combinations.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Withheld coral maximum, restricted training',
          path: ['restricted', 'cells', 'coral/max', 'rmse'],
          format: 'error',
        },
        {
          label: 'Same request, complete training',
          path: ['complete', 'cells', 'coral/max', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Both runs have the same total rows and update budget, but different examples per combination. The chart highlights one withheld task; every cell is checked separately.',
  },
  P072: {
    question: 'Can relflow infer a new rule from examples in the record?',
    modeling:
      'Each record supplies four noisy input–output examples from a newly generated straight line, plus a new query x. Hide the query’s answer and the line parameters. Test rules with a positive slope and intercept together, a combination excluded from training, and compare with a query-only model.',
    comparison: {
      metric: 'RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Query only · withheld rule family',
          path: ['query_only', 'withheld', 'rmse'],
          format: 'number',
        },
        {
          label: 'Visible examples · withheld rule family',
          path: ['context', 'withheld', 'rmse'],
          format: 'number',
        },
      ],
    },
    note: 'No weights are updated for a test record. Query values lie between the supplied examples; this tests a new combination of rules, not extrapolation beyond their x range.',
  },
  P073: {
    question: 'Do reconstruction embeddings help a new small-label task?',
    modeling:
      'Generate three noisy views of a hidden class: two categorical views and a vector. Pretrain an encoder by hiding and reconstructing those views, then freeze it. Give the same nearest-centroid classifier eight labeled examples per class and compare with untrained and shuffled-view encoders.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Untrained encoder',
          path: ['arms', 'untrained', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'Shuffled-view pretraining',
          path: ['arms', 'permuted_views', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'Reconstruction pretraining',
          path: ['arms', 'reconstruction', 'accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The hidden class is never a reconstruction input or target. Every encoder uses the same small labeled probe; all required control margins still count toward the verdict.',
  },
  P074: {
    question: 'Can relflow combine noisy measurements with missing readings?',
    modeling:
      'Generate three noisy sensor readings of one hidden signal and an unrelated fourth reading. Randomly hide sensors during training, excluding one visibility pattern. Compare prediction error from each sensor alone with the error from all three together on the same test records.',
    comparison: {
      metric: 'MSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Sensor 1 alone',
          path: ['patterns', '100', 'mse'],
          format: 'number',
        },
        {
          label: 'Sensor 2 alone',
          path: ['patterns', '010', 'mse'],
          format: 'number',
        },
        {
          label: 'Sensor 3 alone',
          path: ['patterns', '001', 'mse'],
          format: 'number',
        },
        {
          label: 'All three sensors',
          path: ['patterns', '111', 'mse'],
          format: 'number',
        },
      ],
    },
    note: 'The corrupt-sensor stress test is secondary: the model was not trained to detect that corruption. The full verdict also checks missingness patterns and information leakage.',
  },
  P075: {
    question: 'Can relflow follow IDs through larger lists and a second lookup?',
    modeling:
      'Create an ID-keyed source list and ask for a payload through either a direct match or an intermediate link list. IDs and list orders are fresh and independent. In the primary two-hop, four-source test, rotate link IDs while keeping original answers and compare RMSE.',
    comparison: {
      metric: 'RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Two-hop lookup · links changed',
          path: ['hops_2', 'ladder', '4', 'broken', 'rmse'],
          format: 'number',
        },
        {
          label: 'Two-hop lookup · correct links',
          path: ['hops_2', 'ladder', '4', 'intact', 'rmse'],
          format: 'number',
        },
      ],
    },
    note: 'This is one slice of the lookup-size ladder. Every recorded size, direct lookup, and reordering condition has its own gates; the primary slice cannot establish them all.',
  },
  P076: {
    question: 'Can relflow spot an ordinary value in the wrong context?',
    modeling:
      'Generate ordinary numeric answers from a visible group and x, then train the model to reconstruct those answers. Anomalies swap answers between test records, preserving the overall value distribution but breaking context. Compare anomaly AUC with a model trained on shuffled answers.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'AUC',
      conditions: [
        {
          label: 'Shuffled training',
          path: ['shuffled_training', 'anomaly_auc'],
          format: 'auc',
        },
        {
          label: 'Learned context',
          path: ['informative', 'anomaly_auc'],
          format: 'auc',
        },
      ],
    },
    note: 'Anomaly scores are reconstruction errors. Detection thresholds come from clean validation data; normal and valid-tail false alarms remain separate checks.',
  },
  P077: {
    question: 'Should a number be represented by size or percentile?',
    modeling:
      'Give the same numeric tasks to Number and Quantile models. In the highlighted condition, corrupt 2% of training input measurements by multiplying them by 100 without changing their targets; keep evaluation clean. Compare source-unit RMSE on the numeric task.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'Corrupt training, Number',
          path: ['input_tasks', 'amount_number_contaminated', 'rmse'],
          format: 'error',
        },
        {
          label: 'Corrupt training, Quantile',
          path: ['input_tasks', 'amount_quantile_contaminated', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'The chart isolates contaminated inputs. Clean-data, rank, tail, and noisy-target experiments remain below; the test does not presume either type always wins.',
  },
  P078: {
    question: 'Can the archived Cluster model recover stable hidden behavioral groups?',
    modeling:
      'Assign repeated identities to four hidden numerical rules. Train the archived Cluster model to predict fresh observations of those same identities and recover their groups. Compare held-out prediction error with a control where identities have no stable relationship to the rules.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'RMSE',
      conditions: [
        {
          label: 'No persistent groups',
          path: ['no_regime', 'rmse'],
          format: 'error',
        },
        {
          label: 'Stable hidden groups',
          path: ['stable', 'rmse'],
          format: 'error',
        },
      ],
    },
    note: 'Prediction quality alone does not establish group recovery. Separate checks score partition agreement; this does not test unseen identities.',
  },
  P079: {
    question: 'Can one number representation serve additive and multiplicative structure?',
    modeling:
      'Generate additive and multiplicative prediction tasks, including heavy-tailed inputs and targets far from zero. Compare Number and Quantile under matched data and training budgets. The primary chart holds the heavy-tailed task and Number target fixed while changing only the input representation.',
    comparison: {
      kind: 'Controlled comparison',
      metric: 'MAE',
      conditions: [
        {
          label: 'Number inputs · Number target',
          path: ['heavy', 'number_number', 'mae'],
          format: 'error',
        },
        {
          label: 'Quantile inputs · Number target',
          path: ['heavy', 'rank_number', 'mae'],
          format: 'error',
        },
      ],
    },
    note: 'The full checks cover every task. Longer-budget runs and alternative target representations are separate; the primary chart is not a universal ranking.',
    description:
      'Compare Number and Quantile on additive, heavy-tailed, and far-from-zero prediction tasks.',
  },
  P080: {
    question: 'Do reconstruction embeddings separate transaction styles in new records?',
    modeling:
      'Generate three noisy views of four hidden styles, then pretrain an encoder by reconstructing hidden fields without style labels. Freeze the embeddings and fit four clusters on unlabeled training records. Compare test-set group agreement with untrained and shuffled-view encoders and raw features.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Untrained encoder',
          path: ['mean', 'untrained', 'ari'],
          format: 'number',
        },
        {
          label: 'Shuffled-view pretraining',
          path: ['mean', 'permuted', 'ari'],
          format: 'number',
        },
        {
          label: 'Raw features',
          path: ['mean', 'raw', 'ari'],
          format: 'number',
        },
        {
          label: 'Reconstruction pretraining',
          path: ['mean', 'aligned', 'ari'],
          format: 'number',
        },
      ],
    },
    note: 'Four clusters are supplied to the clustering algorithm. Agreement is scored on independent records over three seeds; this does not discover the number of styles.',
  },
  P081: {
    question: 'Do frozen financial embeddings retain their groups with missing evidence?',
    modeling:
      'Generate financial records with noisy amount, frequency, and channel clues to four hidden styles, plus unrelated noise. Fit four clusters on complete training embeddings, then freeze both encoder and centroids. Compare group agreement on the same test records with complete evidence, channel missing, and all useful views missing.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Complete evidence',
          path: ['summary', 'trained', 'complete', 'ari_mean'],
          format: 'number',
        },
        {
          label: 'Channel missing',
          path: ['summary', 'trained', 'one_view_missing', 'ari_mean'],
          format: 'number',
        },
        {
          label: 'All useful views missing',
          path: ['summary', 'trained', 'all_semantic_views_missing', 'ari_mean'],
          format: 'number',
        },
      ],
    },
    note: 'Every condition reuses the same fitted centers. Refitting clusters after removing evidence would ask a different question.',
  },
  P082: {
    question: 'Do transaction histories form meaningful customer neighborhoods?',
    modeling:
      'Generate independent customer histories whose transactions reflect four hidden styles, without customer IDs or style labels in training. Pretrain by reconstructing transaction fields, freeze customer embeddings, and fit four clusters on training histories. Compare held-out group agreement with untrained and mixed-history controls.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Untrained encoder · eight transactions',
          path: ['summary', 'untrained', 'length_8', 'ari_mean'],
          format: 'number',
        },
        {
          label: 'Mixed-history pretraining · eight transactions',
          path: ['summary', 'mixed_histories', 'length_8', 'ari_mean'],
          format: 'number',
        },
        {
          label: 'Reconstruction pretraining · eight transactions',
          path: ['summary', 'reconstruction', 'length_8', 'ari_mean'],
          format: 'number',
        },
      ],
    },
    note: 'The primary chart uses eight transactions. Separate checks evaluate four and twelve, including the length absent from training; four clusters are supplied explicitly.',
  },
  P083: {
    question: 'Can irregular hierarchies preserve which values belong together?',
    modeling:
      'Generate accounts containing days, sessions, and transaction amounts. Pair accounts with the same amounts and collection lengths but different session memberships, then hide the largest session total. Compare familiar and longer structures on independently generated accounts.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Familiar structures',
          path: ['familiar', 'normalized_rmse'],
          format: 'number',
        },
        {
          label: 'Unseen length combinations',
          path: ['unseen_lengths', 'normalized_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'The model receives no prepared totals. An information-bound reference removes grouping; structural-hole checks define null placeholders as zero contribution only for this synthetic task.',
  },
  P084: {
    question: 'Does recent history survive a real collection cap?',
    modeling:
      'Generate histories of at least eight independent amounts and hide the total of the last four. Train matched models that retain either the first four events or the last four under a collection cap. Compare their error on the same full test histories.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Retain the first four events',
          path: ['head', 'familiar', 'normalized_rmse'],
          format: 'number',
        },
        {
          label: 'Retain the last four events',
          path: ['tail', 'familiar', 'normalized_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'First and last windows are disjoint. The head model cannot see any target-defining amount; prefix-invariance checks confirm that discarded records cannot affect the tail model.',
  },
  P085: {
    question: 'Does pretraining reduce the labels needed for a new task?',
    modeling:
      'Generate noisy views of four hidden classes and pretrain without their labels. Adapt the pretrained encoder using eight labeled examples per class, then compare with training from scratch using the same labeled subset and linear classification head.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Train from scratch · eight labels per class',
          path: ['budgets', '8', 'scratch', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'Adapt pretrained encoder · eight labels per class',
          path: ['budgets', '8', 'fine_tuned', 'accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The primary chart uses the smallest label budget. Frozen features, shuffled-view pretraining, raw features, and larger budgets remain separate conditions.',
  },
  P086: {
    question: 'What is forgotten when old examples stop arriving?',
    modeling:
      'Generate three visible cohorts with different numeric response rules. Fit them sequentially, with an independent test set for each cohort. Compare final error with no replay, a small sample of old training rows, and full joint rehearsal.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'No replay of old examples',
          path: ['arms', 'no_replay', 'final_mean_nrmse'],
          format: 'number',
        },
        {
          label: 'Small replay sample',
          path: ['arms', 'small_replay', 'final_mean_nrmse'],
          format: 'number',
        },
        {
          label: 'Joint rehearsal reference',
          path: ['arms', 'joint', 'final_mean_nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'These are different training-data policies under the same update budget. The experiment records forgetting; it does not require replay to win or assume perfect retention.',
  },
  P087: {
    question: 'Can learning preserve small differences beside enormous offsets?',
    modeling:
      'Generate the same small regression differences near zero and near a trillion. Fit matched models using float64 source values, explicit centering, or deliberate float32 input conversion. Compare residual error after removing the target offset.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Input rounded to float32',
          path: ['arms', 'float32', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Large offset retained in float64',
          path: ['arms', 'wide', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Explicitly centered input',
          path: ['arms', 'centered', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The float32 control loses the source differences before the model receives them. This tests preservation and learning of numerical detail, not recovery of information already rounded away.',
  },
  P088: {
    question: 'Can relflow tell a recorded zero from an absent or invalid reading?',
    modeling:
      'Generate three balanced classes identified solely by a numeric zero, null, or valid NaN. Hide the class label. Compare an intact model with separately trained controls that merge NaN into null or replace every input with zero.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Every input filled with zero',
          path: ['arms', 'filled', 'balanced', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'NaN merged into null',
          path: ['arms', 'coalesced', 'balanced', 'accuracy'],
          format: 'percent',
        },
        {
          label: 'All three states preserved',
          path: ['arms', 'intact', 'balanced', 'accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The controls intentionally remove information. Separate recall checks repeat the intact test under changed class proportions so majority accuracy cannot hide a failed state.',
  },
  P089: {
    question: 'Does confidence stay appropriate when useful readings disappear?',
    modeling:
      'Generate yes/no outcomes from a known table of group and signal probabilities. Randomly hide signals during training, then change how often they are missing at test time. Compare written probabilities with the exact answer distribution allowed by each record’s remaining visible information.',
    comparison: {
      metric: 'Probability RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Matched random missingness',
          path: ['conditions', 'matched', 'probability_rmse'],
          format: 'number',
        },
        {
          label: 'More randomly missing readings',
          path: ['conditions', 'missing', 'probability_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'The primary chart compares random missingness conditions. Value-dependent selection is a separate diagnostic because it can change the conditional distribution itself.',
  },
  P090: {
    question: 'Does input jitter buy robustness at a reasonable clean-data cost?',
    modeling:
      'Fit identical regression models on the same small training set with no, modest, or excessive input jitter. Test clean and noisy versions of the same held-out records while keeping targets unchanged. Compare noisy-input error first, then inspect the clean-data cost.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'No jitter · noisy test inputs',
          path: ['arms', 'none', 'conditions', 'strong_noise', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Modest jitter · same noisy inputs',
          path: ['arms', 'small', 'conditions', 'strong_noise', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Excessive jitter · same noisy inputs',
          path: ['arms', 'excessive', 'conditions', 'strong_noise', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'Robustness gains are measured, not required by the pass criteria. Checks met alone does not establish that jitter improved generalization.',
  },
  P091: {
    question: 'Do additional reconstruction objectives interfere with the primary task?',
    modeling:
      'Give a small shared encoder two hidden vector targets: a reliable primary target and a noisy auxiliary target. Compare ordinary summed training with Jacobian-based optimization while holding the actual target values fixed and giving the auxiliary objective eight times the weight.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Summed objectives · primary target',
          path: ['conditions', '8.0', 'summed', 'nrmse', 'primary'],
          format: 'number',
        },
        {
          label: 'Jacobian objectives · primary target',
          path: ['conditions', '8.0', 'jacobian', 'nrmse', 'primary'],
          format: 'number',
        },
      ],
    },
    note: 'This is a separately provisioned research experiment. Finite execution and single-task learnability are gated; superiority of either multi-objective method is not assumed.',
  },
  P092: {
    question: 'Can text meaning transfer to unfamiliar phrases?',
    modeling:
      'Combine a customer request with an ordinary or inverted routing policy and hide a review flag. Refund and cancellation requests change answer when the policy flips. Train on some phrase families and test unseen paraphrases, comparing Text with whole-string Category inputs.',
    comparison: {
      metric: 'Macro F1',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Whole-string Category',
          path: ['arms', 'category', 'macro_f1'],
          format: 'number',
        },
        {
          label: 'Text with routing policy',
          path: ['arms', 'joint', 'macro_f1'],
          format: 'number',
        },
      ],
    },
    note: 'The text encoder uses genuine frozen local weights. This small English routing corpus does not establish general language understanding or operational review policy.',
  },
  P093: {
    question: 'Can a forecast use irregular elapsed time?',
    modeling:
      'Generate four noisy measurements of a straight line at irregular elapsed times, then hide a future value. Compare a model given actual elapsed times with a separately trained control given only ordinal positions and a fixed next-position query.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Positions only',
          path: ['position_only', 'familiar', 'normalized_rmse'],
          format: 'number',
        },
        {
          label: 'Actual elapsed times',
          path: ['elapsed', 'familiar', 'normalized_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'Elapsed times are prepared numeric inputs here, not calendar Period parts. Larger gaps and longer forecast horizons receive separate gates.',
  },
  P094: {
    question: 'Can new requests work on unfamiliar group sizes?',
    modeling:
      'Generate group-specific mean, minimum, and maximum requests. Train with three group–operation combinations withheld and group sizes of one to three. Test those requests on groups of four to six and compare with a model trained on every request combination.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: 'Request combination withheld',
          path: ['restricted_training', 'unseen_lengths', 'unseen_requests', 'normalized_rmse'],
          format: 'number',
        },
        {
          label: 'All request combinations trained',
          path: ['complete_training', 'unseen_lengths', 'unseen_requests', 'normalized_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'The highlighted condition changes both request combinations and group sizes. The full checks cross those shifts separately; familiar performance does not establish the combined case.',
  },
  P096: {
    question: 'Do group-specific thresholds reduce false alarms on noisy groups?',
    modeling:
      'Generate legitimate responses from quiet and noisy groups, then use one fitted predictor to score reconstruction errors. Compare a global anomaly threshold with separate thresholds for each group, all chosen from clean validation data. Measure false alarms on legitimate noisy-group records.',
    comparison: {
      metric: 'False positive rate',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Global threshold · noisy group',
          path: ['global', 'false_positive_by_group', 'noisy'],
          format: 'number',
        },
        {
          label: 'Group-specific threshold · noisy group',
          path: ['conditional', 'false_positive_by_group', 'noisy'],
          format: 'number',
        },
      ],
    },
    note: 'Lower false alarms alone do not establish useful detection. Recall, ranking, and expected precision at 1% anomaly prevalence remain separate measurements.',
  },
  P097: {
    question: 'Can an inferred rule work beyond its examples?',
    modeling:
      'Give each record examples from a fresh straight-line rule and hide its answer at a query x. Train with four examples and queries inside their span. Compare a context model with a query-only control on queries beyond the examples, using the same four-example test records.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Query only · outside example range',
          path: ['query_only', 'count_4/outside', 'normalized_rmse'],
          format: 'number',
        },
        {
          label: 'Four visible examples · outside range',
          path: ['context', 'count_4/outside', 'normalized_rmse'],
          format: 'number',
        },
      ],
    },
    note: 'Two- and eight-example conditions are checked separately. This tests a bounded affine family, not arbitrary rule induction or unlimited extrapolation.',
  },
  P098: {
    question: 'Can lookup handle absent keys and several matches?',
    modeling:
      'Create a fresh query ID with zero, one, two, or three matching source records. Hide both whether any match exists and the average of matching payloads; the synthetic no-match payload is zero. Score match detection separately for every match count.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Recorded conditions',
      conditions: [
        {
          label: '0 matching sources · match detection',
          path: ['intact', 'matches_0', 'found_accuracy'],
          format: 'percent',
        },
        {
          label: '1 matching sources · match detection',
          path: ['intact', 'matches_1', 'found_accuracy'],
          format: 'percent',
        },
        {
          label: '2 matching sources · match detection',
          path: ['intact', 'matches_2', 'found_accuracy'],
          format: 'percent',
        },
        {
          label: '3 matching sources · match detection',
          path: ['intact', 'matches_3', 'found_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'The primary chart shows whether any match was found. Numeric retrieval, record reordering, and broken-key controls must pass their own criteria; this is not an exact database join.',
  },
  P099: {
    question: 'Does Duration learn the same fixed-unit signal as a prepared Number?',
    modeling:
      'Generate explicit UTC timestamp endpoints and a numeric target determined by their signed fractional-day difference. Compare a two-query Duration, a Duration with a configured start, and the exact same elapsed value prepared as Number.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Prepared numeric elapsed time',
          path: ['arms', 'number', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Duration from two endpoints',
          path: ['arms', 'duration', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Duration with configured start',
          path: ['arms', 'static', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'Duration computes elapsed time before learning. This tests equivalent learnable signals, not neural date arithmetic; translating both endpoints is checked separately.',
  },
  P100: {
    question: 'Can ordered outputs fill ranks never shown during training?',
    modeling:
      'Give a noisy numeric clue to an ordered answer and withhold intermediate target ranks during training. Compare Ordinal, rounded Number, and Category on the same unseen integer ranks. All test ranks lie within the observed numeric range.',
    comparison: {
      metric: 'Accuracy',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Category · unseen integer ranks',
          path: ['domains', 'integer', 'category', 'unseen', 'exact_accuracy'],
          format: 'percent',
        },
        {
          label: 'Rounded Number · unseen integer ranks',
          path: ['domains', 'integer', 'number', 'unseen', 'exact_accuracy'],
          format: 'percent',
        },
        {
          label: 'Ordinal · unseen integer ranks',
          path: ['domains', 'integer', 'ordinal', 'unseen', 'exact_accuracy'],
          format: 'percent',
        },
      ],
    },
    note: 'This is interpolation. Declared ordered labels are a separate condition, and Category cannot emit labels absent from its learned output vocabulary.',
  },
  P108: {
    question: 'Can Group recover nonlinear identity regimes?',
    modeling:
      'Assign 100 recurring identities to five hidden nonlinear response curves. Give identity and x, hide y, and learn Group assignments through prediction. Compare final assignment agreement for stable identities with a control whose identities are shuffled across rules.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'No persistent identity–rule link',
          path: ['group_counts', '5', 'no_regime', 'adjusted_rand'],
          format: 'number',
        },
        {
          label: 'Stable hidden rules',
          path: ['group_counts', '5', 'stable', 'adjusted_rand'],
          format: 'number',
        },
      ],
    },
    note: 'The five Group slots are fixed in advance. Evaluation uses fresh observations of known identities, not unseen identities or discovery of the number of groups.',
  },
  P109: {
    question: 'Can a plain Group recover label groups?',
    modeling:
      'Generate five label groups with ten recurring identities each, plus unrelated context. Train a sibling label predictor while leaving the Group state unmasked. Score whether the final assignments recover the hidden label groups.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Recorded measurement',
      conditions: [
        {
          label: 'Final assignment agreement',
          path: ['adjusted_rand', -1],
          format: 'number',
        },
      ],
    },
    note: 'Assignments learn from the downstream label objective. This is a current Group hypothesis, not a reused measurement from the retired Cluster experiment; held-out label and shuffled-ID accuracy are separate gates.',
  },
  P110: {
    question: 'Can a masked Group recover label groups?',
    modeling:
      'Generate the same five-group label task with recurring identities and unrelated context, and add masking of the Group state while training the sibling label target. Score final assignment agreement with the generating groups.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Recorded measurement',
      conditions: [
        {
          label: 'Final assignment agreement',
          path: ['adjusted_rand', -1],
          format: 'number',
        },
      ],
    },
    note: 'The state mask trains the state head; downstream label supervision trains assignments. P109 tests the unmasked configuration separately, and old Cluster outcomes are not reused.',
  },
  P111: {
    question: 'Can Group recover known hidden regimes?',
    modeling:
      'Assign 64 recurring identities to four hidden linear response rules. Hide each new numeric answer and learn assignments into four fixed Group slots. Compare agreement with the true partition against a no-regime control that preserves numeric records but shuffles which identity accompanies each rule.',
    comparison: {
      metric: 'Adjusted Rand index',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'No persistent identity–rule link',
          path: ['group_counts', '4', 'no_regime', 'adjusted_rand'],
          format: 'number',
        },
        {
          label: 'Stable hidden rules',
          path: ['group_counts', '4', 'stable', 'adjusted_rand'],
          format: 'number',
        },
      ],
    },
    note: 'The model is supplied four slots. This tests recovery for known identities, not adaptive group-count selection or generalization to new identities.',
  },
  P112: {
    question: 'When does grouping identities help prediction?',
    modeling:
      'Generate known identities belonging to four hidden response groups. Train Group and Category models on matched dense, sparse, and skewed identity populations, then score all identities equally. The primary comparison uses the dense condition and the same model width and update budget.',
    comparison: {
      metric: 'Normalized RMSE',
      kind: 'Controlled comparison',
      conditions: [
        {
          label: 'Category · dense identities',
          path: ['conditions', 'dense', 'arms', 'category', 'nrmse'],
          format: 'number',
        },
        {
          label: 'Group · dense identities',
          path: ['conditions', 'dense', 'arms', 'group', 'nrmse'],
          format: 'number',
        },
      ],
    },
    note: 'The true-group reference receives information hidden from the other models. Sparse and rare-identity gains are diagnostics; checks met does not imply Group always beats Category.',
  },
  P101: {
    question: 'Which numeric shifts trigger distribution warnings?',
    modeling:
      'Train on Gaussian values and observe independent held-out populations with shifted location, scale, or shape. Compare those interventions with same-population controls and a correlation change that preserves the marginal distributions.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Declared shifts detected',
          path: ['detection_rate'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
  P102: {
    question: 'Which typed input shifts trigger distribution warnings?',
    modeling:
      'Train on categories, Boolean values, sets, vectors, calendar parts, and nested numeric inputs. Change one declared feature at a time in matched held-out records and retain independent same-population controls.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Declared shifts detected',
          path: ['detection_rate'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
  P103: {
    question: 'Do drift policies and checkpoints preserve monitoring state?',
    modeling:
      'Apply the same numeric shift to monitored, tolerant, and disabled inputs after training. Check minimum counts, warning suppression, clearing, and checkpoint restoration with three fixed seeds.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Checks met',
      conditions: [
        {
          label: 'Monitoring contracts met',
          path: ['contracts_met'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
  P104: {
    question: 'Does Enum detect changes among declared labels?',
    modeling:
      'Train with four declared labels and independent Boolean targets. Measure frequency and null-rate shifts across validation, test, and prediction, with same-population and folded-spelling controls. Check that undeclared labels remain rejected.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Declared shifts detected',
          path: ['detection_rate'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
  P105: {
    question: 'Does Hash detect identity population changes?',
    modeling:
      'Train on recurring identifiers and independent Boolean targets. Change identity frequencies, introduce new identities, or change null rates across validation, test, and prediction. Retain matched controls and check that profiling remains independent of training salts.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Declared shifts detected',
          path: ['detection_rate'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
  P113: {
    question: 'Does Group detect input drift while preserving assignments?',
    modeling:
      'Train Group inputs with an independent Boolean target, then change identity frequencies, introduce identities, or change null rates across validation, test, and prediction. Compare with matched controls and check that evaluation preserves learned vocabulary and assignments.',
    comparison: {
      kind: 'Recorded measurement',
      metric: 'Proportion',
      conditions: [
        {
          label: 'Declared shifts detected',
          path: ['detection_rate'],
          format: 'number',
        },
      ],
    },
    note: 'Three fixed seeds exercise a short real optimizer loop. These are rank-local marginal monitoring checks; they do not establish predictive accuracy, calibrated false-positive rates, or joint-distribution detection. Quiet controls and every predeclared check remain in the complete record.',
  },
};
