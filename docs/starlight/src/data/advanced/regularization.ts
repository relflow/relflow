import type { Concept } from '../concepts/types';
import { scenarios } from '../concepts/scenarios';

const examples: Record<(typeof scenarios)[number]['id'], Omit<Concept['examples'][number], 'caseId'>> = {
  tickets: {
    title: 'Vary the conversation context, keep the answers hidden',
    description: 'This alternative TicketModel schema adds occasional input masking, attention/feed-forward dropout, and sampled conversation windows. Its inherited escalation and resolution targets stay masked.',
    blocks: [{
      label: 'An alternative schema for a new training run', lang: 'python',
      code: `class RegularizedMessage(TicketMessage):
    """A message whose timing input may be hidden during training."""

    minutes_since_previous = rf.Number(mask=rf.Mask(rate=0.1))


class RegularizedTicket(TicketModel):
    """A ticket learned from varied conversation contexts."""

    messages = RegularizedMessage(
        length=64, overflow="random", dropout=0.1,
    )`,
      note: 'These classes extend the shared class declarations. A source must supply more than 64 messages for the retained window to vary. Bound that candidate pool separately.',
    }],
    takeaway: 'Randomly hiding an input is a robustness choice. The inherited mask=True targets remain always hidden and supervised.',
  },
  churn: {
    title: 'Perturb usage without changing the cancellation label',
    description: 'Keep the existing churn target and try small multiplicative noise on observed usage. Sample a contiguous usage window only when the supplied history exceeds its cap.',
    blocks: [{
      label: 'Usage noise and an independently masked support history', lang: 'python',
      code: `class RegularizedUsage(MonthlyUsage):
    """A usage month with training-only noise on continuous inputs."""

    minutes = rf.Number(jitter=rf.Jitter(multiply=0.05, normalize=False))
    data_gb = rf.Number(jitter=rf.Jitter(multiply=0.05, normalize=False))


class RegularizedChurn(ChurnModel):
    """A subscriber modeled with varied usage and support context."""

    monthly_usage = RegularizedUsage(length=12, overflow="random")
    support_tickets = SupportTicket(length=32, overflow="tail", mask=0.1)`,
      note: 'normalize=False puts this multiplicative jitter on the raw numeric representation. The source values and churn_90d answers are unchanged; a float mask adds training-only input dropout.',
    }],
    takeaway: 'Usage noise, sampled months, and hidden support context are separate choices. Compare them on a held-out subscriber cohort.',
  },
  demand: {
    title: 'Augment sales history, preserve every forecast slot',
    description: 'Vary the observed sales context while keeping the supplied future weeks intact. The inherited horizon still has length=12, overflow="error", and masked units_sold targets.',
    blocks: [{
      label: 'Only the observed history gets a random window', lang: 'python',
      code: `class RegularizedSalesWeek(SalesWeek):
    """An observed sales week with training-only magnitude noise."""

    units_sold = rf.Number(
        jitter=rf.Jitter(multiply=0.03, normalize=False),
    )


class RegularizedDemand(DemandModel):
    """A demand model that varies past context, preserving its horizon."""

    history = RegularizedSalesWeek(
        length=104, overflow="random", dropout=0.1,
    )`,
      note: 'Random overflow samples a contiguous training window from histories longer than 104 rows; validation and prediction keep the tail. It is not a rule for generating or dropping future weeks.',
    }],
    takeaway: 'A history window changes the evidence for a forecast. Cropping the horizon would change which forecasts are requested.',
  },
  credit: {
    title: 'Test robustness to incomplete account attributes',
    description: 'Occasionally hide an account’s missed-payment count and perturb its balance representation during training. Keep the account collection and the default target contract intact.',
    blocks: [{
      label: 'Regularize account inputs without discarding accounts', lang: 'python',
      code: `class RegularizedAccount(CreditAccount):
    """A credit account with training-only input perturbations."""

    balance = rf.Number(jitter=rf.Jitter(add=0.02))
    missed_payments = rf.Number(mask=0.05)


class RegularizedCredit(CreditModel):
    """An application modeled with regularized account context."""

    accounts = RegularizedAccount(
        length=64, overflow="error", dropout=0.1,
    )`,
      note: 'Jitter defaults to normalize=True, so add=0.02 is a bound in standardized representation units, not currency units. These illustrative rates need validation on the intended population.',
    }],
    takeaway: 'Keep the account set intact while testing sensitivity to individual attributes. Validate the added perturbations against the same held-out application records.',
  },
  maintenance: {
    title: 'Vary sensor windows and continuous readings',
    description: 'Try modest sensor jitter and contiguous windows of earlier observations. Every retained sensor row keeps its hour, vibration, temperature, and load together.',
    blocks: [{
      label: 'Apply noise at the sensor-reading boundary', lang: 'python',
      code: `class RegularizedReading(SensorReading):
    """A sensor observation with training-only continuous noise."""

    vibration = rf.Number(jitter=rf.Jitter(add=0.02))
    temperature = rf.Number(jitter=rf.Jitter(add=0.02))


class RegularizedMaintenance(MaintenanceModel):
    """An asset modeled from varied historical sensor windows."""

    sensor_readings = RegularizedReading(
        length=168, overflow="random", dropout=0.1,
    )`,
      note: 'The candidate pool must contain only readings available at the prediction cutoff. A random window cannot repair future-data leakage, and the small shared illustration does not exceed this cap.',
    }],
    takeaway: 'Windowing preserves aligned records. Jitter changes continuous training representations; neither changes the failure_30d label.',
  },
  claims: {
    title: 'Regularize invoice context, preserve submitted claims',
    description: 'Perturb billed-amount representations inside the nested invoice context. Every claim remains present with its own payout and fraud targets; no target-bearing collection is randomly cropped.',
    blocks: [{
      label: 'Configure the nested input path through subclasses', lang: 'python',
      code: `class RegularizedLineItem(LineItem):
    """A billed item with training-only amount noise."""

    amount = rf.Number(jitter=rf.Jitter(multiply=0.02, normalize=False))


class RegularizedInvoice(Invoice):
    """An invoice whose item encoder uses dropout."""

    line_items = RegularizedLineItem(length=32, overflow="error", dropout=0.1)


class RegularizedClaim(Claim):
    """A claim retaining every supplied invoice and target."""

    invoices = RegularizedInvoice(length=16, overflow="error")


class RegularizedClaims(ClaimsModel):
    """A policy with regularized evidence for each submitted claim."""

    claims = RegularizedClaim(length=32, overflow="error")`,
      note: 'The input perturbation leaves recorded invoice amounts and training answers unchanged. An over-cap collection raises instead of silently dropping claims, invoices, or items.',
    }],
    takeaway: 'Keep the prediction request stable. Regularize the context used to answer each claim, rather than changing the submitted claim list.',
  },
};

export const regularization: Concept = {
  id: 'regularization',
  title: 'Regularization',
  description: 'Separate supervised target hiding from optional input masking, numeric jitter, neural dropout, and sampled history windows.',
  intro: [
    'A schema specifies what the model can learn from. Regularization varies part of that training experience so the model has less opportunity to depend on one exact input or context.',
    'The examples extend the original class schemas for a new training run. Their rates illustrate configuration choices; they are not measured recommendations. Compare changes against an unchanged baseline on held-out data.',
  ],
  principles: [
    { title: 'Keep target hiding permanent', text: 'mask=True is the supervised-target contract: that field is never embedded and receives a reconstruction objective. A float mask or Mask(rate=...) instead adds training-only random input dropout by default. Replacing a target’s mask=True with a float changes its role.' },
    { title: 'Choose the boundary for noise', text: 'Jitter perturbs supported continuous training representations, while dropout configures neural attention/feed-forward regularization. Jitter uses standardized Number values by default; normalize=False selects its raw representation. Neither edits source records or target values.' },
    { title: 'Sample history without changing the question', text: 'A bounded history with overflow="random" uses a contiguous random training window and retains the tail for validation, test, and prediction. Supply more than the cap for sampling to vary and bound the candidate pool separately' },
  ],
  examples: scenarios.map(({ id }) => ({ caseId: id, ...examples[id] })),
  takeaway: 'Target masks define the task. Input masking, jitter, dropout, and history sampling are optional training choices that need their own held-out comparison.',
  links: [
    { label: 'Advanced masking and pretraining', href: '/advanced/masking-and-pretraining/' },
    { label: 'Mask configuration', href: '/api-map/#api-rf-mask' },
    { label: 'Jitter configuration', href: '/api-map/#api-rf-jitter' },
    { label: 'Number inputs and outputs', href: '/data-types/number/' },
    { label: 'Controlled learning experiments', href: '/evidence/' },
  ],
};
