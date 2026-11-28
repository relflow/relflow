import { prediction, type TypeExample } from '../type-example.ts';

export const periodExamples: TypeExample[] = [
  {
    id: 'support',
    label: 'Recurring patterns',
    summary: 'Estimate response time',
    title: 'How long might this support request wait?',
    description: 'Combine the weekday and arrival hour with a separate elapsed-time measurement.',
    dataset: {
      id: 'support', label: 'Support request arrival',
      observation: 'One support request at arrival.',
      description: 'Calendar timing and elapsed time since the previous request provide context for a response-time target.',
      entities: [{ id: 'request', name: 'Support request', description: 'Known arrival context with the eventual response-time label.', columns: [
        { name: 'received_at', type: 'datetime', description: 'Arrival timestamp supplied to Period for its weekday and hour. The example’s timestamp string is interpreted in UTC.' },
        { name: 'minutes_since_previous_request', type: 'number', description: 'Minutes elapsed since the preceding request, known at arrival. This is elapsed time, separate from the calendar cycles.' },
        { name: 'response_minutes', type: 'number', role: 'target', description: 'Observed response time in minutes for a historical request. Omitted from new prediction inputs.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class SupportRequest(rf.Model):
    """A support request with its arrival and response time."""

    received_at = rf.Period("day_of_week", "hour_of_day")
    minutes_since_previous_request = rf.Number
    response_minutes = rf.Number(predict=True)`,
    inputs: { received_at: '2026-11-13T16:30:00Z', minutes_since_previous_request: 3 },
    answers: { response_minutes: 18 },
    output: { predictions: { response_minutes: prediction(20.4) } },
    outputNote: 'The prediction is a duration in minutes from Number. Period supplies recurring calendar context and produces no datetime output.',
    takeaway: 'Friday afternoon and three minutes since the previous request describe different information. Period captures the calendar cycles; Number captures the elapsed time.',
  },
  {
    id: 'forecast-slots',
    label: 'Dated forecast slots',
    summary: 'Predict for supplied dates',
    title: 'How many units might sell on each planned date?',
    description: 'Supply the future dates you want to score, with known promotions and recent sales as context.',
    dataset: {
      id: 'forecast-slots', label: 'Dated sales forecast',
      observation: 'One product and the future dates selected for forecasting.',
      description: 'Known future dates and promotions remain attached to each supplied daily sales target.',
      entities: [
        { id: 'forecast', name: 'Sales forecast', description: 'Product context with the complete requested forecast horizon.', columns: [
          { name: 'product', type: 'string', description: 'Product being forecast, such as notebook.' },
          { name: 'units_sold_last_7d', type: 'number', description: 'Observed units sold during the seven days before the forecast cutoff.' },
          { name: 'days', type: 'records[]', role: 'collection', description: 'Future dates chosen by the caller. The model returns a sales estimate for each retained record.' },
        ] },
        { id: 'day', name: 'Forecast day', description: 'One supplied future date in days.', columns: [
          { name: 'date', type: 'date', description: 'Known forecast date. Period uses its month and weekday as calendar context.' },
          { name: 'promotion', type: 'boolean', description: 'Promotion plan for this date, known when the forecast is made.' },
          { name: 'units_sold', type: 'number', role: 'target', description: 'Units eventually sold on this date, used as a historical training label and omitted from new forecast requests.' },
        ] },
      ],
      relations: [{ from: 'forecast', to: 'day', fromColumn: 'days', cardinality: 'many', description: 'Each product forecast contains the dates supplied for per-day predictions.' }],
    },
    schema: `import relflow as rf


class ForecastDay(rf.Branch):
    """A supplied date with a daily sales target."""

    date = rf.Period("month_of_year", "day_of_week")
    promotion = rf.Boolean
    units_sold = rf.Number(predict=True)


class SalesForecast(rf.Model):
    """A product and the future dates selected for forecasting."""

    product = rf.Category
    units_sold_last_7d = rf.Number
    days = ForecastDay(length=14, overflow="error")`,
    inputs: {
      product: 'notebook',
      units_sold_last_7d: 105,
      days: [
        { date: '2026-11-14', promotion: false },
        { date: '2026-11-15', promotion: true },
      ],
    },
    answers: { days: [{ units_sold: 12 }, { units_sold: 24 }] },
    output: {
      predictions: {
        days: [
          { units_sold: prediction(13.2) },
          { units_sold: prediction(22.6) },
        ],
      },
    },
    outputNote: 'The two numeric answers correspond to the two supplied dates. The model predicts sales at these positions; it does not generate dates or a new collection.',
    takeaway: 'Period can describe a known future calendar slot. Training records pair those same date fields with observed sales; prediction requests omit the sales answers.',
  },
  {
    id: 'login',
    label: 'Calendar and recency',
    summary: 'Review a login event',
    title: 'Does this login need an additional check?',
    description: 'Use a login’s recurring calendar position alongside account and recent-activity context.',
    dataset: {
      id: 'login', label: 'Login review',
      observation: 'One login event when deciding whether to require a check.',
      description: 'Known calendar, recency, and device context teach a Boolean security-check decision.',
      entities: [{ id: 'login', name: 'Login', description: 'The event context and historical check label.', columns: [
        { name: 'occurred_at', type: 'datetime', description: 'Login timestamp supplied to Period for weekday and hour. The example’s timestamp string expresses a UTC instant.' },
        { name: 'account_age_days', type: 'number', description: 'Account age in days when the login occurs.' },
        { name: 'minutes_since_last_login', type: 'number', description: 'Minutes elapsed since the preceding login, known at this event.' },
        { name: 'new_device', type: 'boolean', description: 'Whether this login uses a device considered new at the decision time.' },
        { name: 'needs_check', type: 'boolean', role: 'target', description: 'Historical label indicating whether an additional check is needed. Omitted when scoring new logins.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class Login(rf.Model):
    """A login event considered for an additional security check."""

    occurred_at = rf.Period("day_of_week", "hour_of_day")
    account_age_days = rf.Number
    minutes_since_last_login = rf.Number
    new_device = rf.Boolean
    needs_check = rf.Boolean(predict=True)`,
    inputs: {
      occurred_at: '2026-11-14T02:15:00Z',
      account_age_days: 420,
      minutes_since_last_login: 720,
      new_device: true,
    },
    answers: { needs_check: true },
    output: { predictions: { needs_check: prediction({ probability: 0.76 }) } },
    outputNote: 'The Boolean target returns its probability of true. The timestamp is an input; neither its text nor a decoded calendar value is included in this prediction row.',
    takeaway: 'These strings express UTC hours. For local-hour patterns, prepare an Arrow timestamp column with the intended time zone; offset-bearing strings are parsed as UTC.',
  },
];
