// Synthetic unit data only; these values are never published as learning measurements.
export function record(
  metrics = { measured: { nrmse: 0.2, rmse: 0.3, baseline_rmse: 1 }, permutation_drift: 0.01 },
) {
  return {
    mode: 'full',
    outcome: 'met',
    checks: { 'Fixture signal': true, 'Fixture control': true },
    metrics,
    provenance: {},
    seed: 7,
    started_at: '2026-01-01T00:00:00Z',
    duration_seconds: 1,
    accelerator: 'cpu',
    threads: 1,
    steps_override: null,
  };
}
