import type { Dataset } from '../datasets/types';

/** Illustrative source tables for the six example datasets. */
export const datasets: Dataset[] = [
  {
    id: 'tickets',
    label: 'Support tickets',
    observation: 'One support ticket at triage.',
    description: 'Support tickets record the product, triage priority, conversation messages, escalation outcome, and time to resolution.',
    entities: [
      {
        id: 'ticket', name: 'Ticket',
        description: 'One support request, with its triage details and recorded resolution outcomes.',
        columns: [
          { name: 'ticket_id', type: 'string', role: 'key', description: 'Unique identifier linking a ticket to its conversation messages.' },
          { name: 'product', type: 'string', description: 'Product associated with the support request, such as analytics.' },
          { name: 'priority', type: 'string', description: 'Priority recorded at triage: low, normal, or high.' },
          { name: 'escalated', type: 'boolean', role: 'target', description: 'Whether the ticket was escalated after triage.' },
          { name: 'hours_to_resolve', type: 'number', role: 'target', description: 'Recorded time to resolution, in hours.' },
        ],
      },
      {
        id: 'message', name: 'Message',
        description: 'One message in a ticket’s conversation, ordered from earlier to later messages.',
        columns: [
          { name: 'message_id', type: 'string', role: 'key', description: 'Unique identifier for one conversation message.' },
          { name: 'ticket_id', type: 'string', role: 'reference', description: 'Identifier of the ticket this message belongs to.' },
          { name: 'author', type: 'string', description: 'Author role for this message, such as customer or agent.' },
          { name: 'body', type: 'string', description: 'Message text available at triage.' },
          { name: 'minutes_since_previous', type: 'number', description: 'Elapsed minutes since the preceding message in the conversation.' },
        ],
      },
    ],
    relations: [
      { from: 'ticket', to: 'message', fromColumn: 'ticket_id', toColumn: 'ticket_id', description: 'One ticket has zero or more messages available at triage.' },
    ],
  },
  {
    id: 'churn',
    label: 'Customer churn',
    observation: 'One subscriber at a cutoff date.',
    description: 'Subscriber records link to monthly usage and support histories, with cancellation outcomes recorded for the following 90 days.',
    entities: [
      {
        id: 'subscriber', name: 'Subscriber',
        description: 'One subscriber’s plan, tenure, and recorded cancellation outcome.',
        columns: [
          { name: 'subscriber_id', type: 'string', role: 'key', description: 'Unique identifier linking a subscriber to monthly usage and support records.' },
          { name: 'plan', type: 'string', description: 'Subscription plan known at the cutoff, such as pro.' },
          { name: 'tenure_months', type: 'number', description: 'Subscriber tenure in months at the cutoff.' },
          { name: 'churn_90d', type: 'boolean', role: 'target', description: 'Whether the subscriber canceled within 90 days after the cutoff.' },
        ],
      },
      {
        id: 'monthly_usage', name: 'Monthly usage',
        description: 'One month of a subscriber’s usage, ordered from earlier to later months.',
        columns: [
          { name: 'usage_id', type: 'string', role: 'key', description: 'Unique identifier for one monthly usage record.' },
          { name: 'subscriber_id', type: 'string', role: 'reference', description: 'Identifier of the subscriber associated with this monthly usage record.' },
          { name: 'month', type: 'date', description: 'Date identifying the observed month.' },
          { name: 'minutes', type: 'number', description: 'Usage minutes recorded for this month.' },
          { name: 'data_gb', type: 'number', description: 'Data usage in gigabytes for this month.' },
          { name: 'overage_fee', type: 'number', description: 'Overage charge recorded for this month, in the dataset’s currency units.' },
        ],
      },
      {
        id: 'support_ticket', name: 'Support ticket',
        description: 'One subscriber support ticket, with its status recorded at the cutoff.',
        columns: [
          { name: 'ticket_id', type: 'string', role: 'key', description: 'Unique identifier for one historical support ticket.' },
          { name: 'subscriber_id', type: 'string', role: 'reference', description: 'Identifier of the subscriber associated with this support ticket.' },
          { name: 'channel', type: 'string', description: 'Support channel used for this ticket, such as chat.' },
          { name: 'resolved', type: 'boolean', description: 'Whether the support ticket was resolved by the subscriber cutoff.' },
          { name: 'days_open', type: 'number', description: 'Recorded time the ticket had been open, in days, at the cutoff.' },
        ],
      },
    ],
    relations: [
      { from: 'subscriber', to: 'monthly_usage', fromColumn: 'subscriber_id', toColumn: 'subscriber_id', description: 'One subscriber has zero or more monthly usage records.' },
      { from: 'subscriber', to: 'support_ticket', fromColumn: 'subscriber_id', toColumn: 'subscriber_id', description: 'One subscriber has zero or more historical support tickets.' },
    ],
  },
  {
    id: 'demand',
    label: 'Demand forecasting',
    observation: 'One product at one store, with past and future weeks.',
    description: 'Store and product records link to weekly sales history and future weeks, including promotion plans and eventual sales totals.',
    entities: [
      {
        id: 'store_product', name: 'Store product',
        description: 'One store-and-product pair, linked to its past and future weekly records.',
        columns: [
          { name: 'store_product_id', type: 'string', role: 'key', description: 'Unique identifier linking a store-and-product pair to its weekly records.' },
          { name: 'store', type: 'string', description: 'Store identifier, such as store-4.' },
          { name: 'sku', type: 'string', description: 'Product identifier, such as boots-9.' },
        ],
      },
      {
        id: 'sales_week', name: 'Sales week',
        description: 'One past sales week for a store-and-product pair, ordered from earlier to later weeks.',
        columns: [
          { name: 'sales_week_id', type: 'string', role: 'key', description: 'Unique identifier for one weekly sales record.' },
          { name: 'store_product_id', type: 'string', role: 'reference', description: 'Identifier of the store-and-product pair associated with this sales week.' },
          { name: 'week', type: 'date', description: 'Date identifying the past sales week.' },
          { name: 'units_sold', type: 'number', description: 'Number of units sold during this past week.' },
          { name: 'promotion', type: 'boolean', description: 'Whether a promotion applied during this past week.' },
          { name: 'stockout', type: 'boolean', description: 'Whether a stockout was recorded during this past week.' },
        ],
      },
      {
        id: 'forecast_week', name: 'Forecast week',
        description: 'One future week for a store-and-product pair, with its promotion plan and eventual sales total.',
        columns: [
          { name: 'forecast_week_id', type: 'string', role: 'key', description: 'Unique identifier for one future weekly record.' },
          { name: 'store_product_id', type: 'string', role: 'reference', description: 'Identifier of the store-and-product pair associated with this future week.' },
          { name: 'week', type: 'date', description: 'Date identifying the future week.' },
          { name: 'promotion', type: 'boolean', description: 'Whether a promotion is planned for this future week.' },
          { name: 'units_sold', type: 'number', role: 'target', description: 'Number of units eventually sold during this future week.' },
        ],
      },
    ],
    relations: [
      { from: 'store_product', to: 'sales_week', fromColumn: 'store_product_id', toColumn: 'store_product_id', description: 'One store-and-product pair has zero or more past weekly sales records.' },
      { from: 'store_product', to: 'forecast_week', fromColumn: 'store_product_id', toColumn: 'store_product_id', description: 'One store-and-product pair has zero or more future weekly records.' },
    ],
  },
  {
    id: 'credit',
    label: 'Credit risk',
    observation: 'One credit application at assessment.',
    description: 'Credit applications record requested terms, existing account snapshots, recent inquiries, and defaults recorded over the following 12 months.',
    entities: [
      {
        id: 'application', name: 'Application',
        description: 'One credit application at assessment, with its requested terms and subsequent default outcome.',
        columns: [
          { name: 'application_id', type: 'string', role: 'key', description: 'Unique identifier linking an application snapshot to its account and inquiry records.' },
          { name: 'product', type: 'string', description: 'Requested credit product, such as installment.' },
          { name: 'requested_amount', type: 'number', description: 'Amount requested in the application, in the dataset’s currency units.' },
          { name: 'default_12m', type: 'boolean', role: 'target', description: 'Whether default occurred within 12 months after assessment.' },
        ],
      },
      {
        id: 'account', name: 'Credit account',
        description: 'One existing account in an application’s credit snapshot.',
        columns: [
          { name: 'account_id', type: 'string', role: 'key', description: 'Unique identifier for one account record in this application snapshot.' },
          { name: 'application_id', type: 'string', role: 'reference', description: 'Identifier of the application associated with this account snapshot.' },
          { name: 'account_type', type: 'string', description: 'Type of existing credit account, such as card.' },
          { name: 'balance', type: 'number', description: 'Account balance known at assessment, in the dataset’s currency units.' },
          { name: 'months_open', type: 'number', description: 'Age of this account in months at assessment.' },
          { name: 'missed_payments', type: 'number', description: 'Recorded missed-payment count available at assessment.' },
        ],
      },
      {
        id: 'inquiry', name: 'Credit inquiry',
        description: 'One recent inquiry in an application’s credit snapshot, ordered from earlier to later inquiries.',
        columns: [
          { name: 'inquiry_id', type: 'string', role: 'key', description: 'Unique identifier for one inquiry in this application snapshot.' },
          { name: 'application_id', type: 'string', role: 'reference', description: 'Identifier of the application associated with this credit inquiry.' },
          { name: 'lender', type: 'string', description: 'Lender associated with the inquiry.' },
          { name: 'age_days', type: 'number', description: 'Days elapsed between the inquiry and assessment.' },
        ],
      },
    ],
    relations: [
      { from: 'application', to: 'account', fromColumn: 'application_id', toColumn: 'application_id', description: 'One application snapshot contains zero or more existing account records.' },
      { from: 'application', to: 'inquiry', fromColumn: 'application_id', toColumn: 'application_id', description: 'One application snapshot contains zero or more recent credit inquiries.' },
    ],
  },
  {
    id: 'maintenance',
    label: 'Predictive maintenance',
    observation: 'One asset at a cutoff date.',
    description: 'Asset records link to sensor readings and completed maintenance jobs, with failures recorded over the following 30 days.',
    entities: [
      {
        id: 'asset', name: 'Asset',
        description: 'One asset, with its type and age at the cutoff and subsequent failure outcome.',
        columns: [
          { name: 'asset_id', type: 'string', role: 'key', description: 'Unique identifier linking an asset to its sensor readings and maintenance jobs.' },
          { name: 'asset_type', type: 'string', description: 'Type of equipment, such as pump.' },
          { name: 'age_years', type: 'number', description: 'Asset age in years at the cutoff.' },
          { name: 'failure_30d', type: 'boolean', role: 'target', description: 'Whether the asset failed within 30 days after the cutoff.' },
        ],
      },
      {
        id: 'sensor_reading', name: 'Sensor reading',
        description: 'One timestamped sensor reading for an asset, ordered from earlier to later readings.',
        columns: [
          { name: 'reading_id', type: 'string', role: 'key', description: 'Unique identifier for one sensor reading.' },
          { name: 'asset_id', type: 'string', role: 'reference', description: 'Identifier of the asset associated with this sensor reading.' },
          { name: 'hour', type: 'datetime', description: 'Timestamp of the sensor reading.' },
          { name: 'vibration', type: 'number', description: 'Vibration measurement in the source dataset’s units.' },
          { name: 'temperature', type: 'number', description: 'Temperature measurement in the source dataset’s units.' },
          { name: 'load', type: 'number', description: 'Equipment load recorded with this reading, using the source dataset’s scale.' },
        ],
      },
      {
        id: 'work_order', name: 'Work order',
        description: 'One maintenance job completed by the cutoff.',
        columns: [
          { name: 'work_order_id', type: 'string', role: 'key', description: 'Unique identifier for one completed maintenance job.' },
          { name: 'asset_id', type: 'string', role: 'reference', description: 'Identifier of the asset associated with this maintenance job.' },
          { name: 'type', type: 'string', description: 'Type of completed maintenance work, such as inspection.' },
          { name: 'parts_replaced', type: 'string[]', description: 'Labels of parts replaced during this job; an empty list means no parts were replaced.' },
          { name: 'age_days', type: 'number', description: 'Days elapsed between the completed job and the cutoff.' },
        ],
      },
    ],
    relations: [
      { from: 'asset', to: 'sensor_reading', fromColumn: 'asset_id', toColumn: 'asset_id', description: 'One asset has zero or more sensor readings.' },
      { from: 'asset', to: 'work_order', fromColumn: 'asset_id', toColumn: 'asset_id', description: 'One asset has zero or more completed maintenance jobs.' },
    ],
  },
  {
    id: 'claims',
    label: 'Insurance claims',
    observation: 'One policy with submitted claims at review.',
    description: 'Policies link to submitted claims, supporting invoices, and billed items. Each claim includes its final payout and recorded fraud outcome.',
    entities: [
      {
        id: 'policy', name: 'Policy',
        description: 'One insurance policy with submitted claims under review.',
        columns: [
          { name: 'policy_id', type: 'string', role: 'key', description: 'Unique identifier linking a policy to its submitted claims.' },
          { name: 'policy_type', type: 'string', description: 'Type of insurance policy, such as auto.' },
          { name: 'coverage_limit', type: 'number', description: 'Policy coverage limit known at review, in the dataset’s currency units.' },
        ],
      },
      {
        id: 'claim', name: 'Claim',
        description: 'One submitted claim, with its reported incident details, final payout, and fraud outcome.',
        columns: [
          { name: 'claim_id', type: 'string', role: 'key', description: 'Unique identifier linking a claim to its supporting invoices.' },
          { name: 'policy_id', type: 'string', role: 'reference', description: 'Identifier of the policy associated with this claim.' },
          { name: 'cause', type: 'string', description: 'Reported cause of this claim, such as collision.' },
          { name: 'reported_after_days', type: 'number', description: 'Days between the incident and the claim being reported.' },
          { name: 'paid_amount', type: 'number', role: 'target', description: 'Final amount paid for this claim, in the dataset’s currency units.' },
          { name: 'is_fraudulent', type: 'boolean', role: 'target', description: 'Whether the claim was recorded as fraudulent.' },
        ],
      },
      {
        id: 'invoice', name: 'Invoice',
        description: 'One invoice supporting a claim, with its own billed line items.',
        columns: [
          { name: 'invoice_id', type: 'string', role: 'key', description: 'Unique identifier linking an invoice to its billed line items.' },
          { name: 'claim_id', type: 'string', role: 'reference', description: 'Identifier of the claim associated with this invoice.' },
          { name: 'provider', type: 'string', description: 'Identifier of the provider named on this invoice.' },
        ],
      },
      {
        id: 'line_item', name: 'Line item',
        description: 'One billed line item belonging to an invoice.',
        columns: [
          { name: 'line_item_id', type: 'string', role: 'key', description: 'Unique identifier for one billed line item.' },
          { name: 'invoice_id', type: 'string', role: 'reference', description: 'Identifier of the invoice associated with this billed line item.' },
          { name: 'code', type: 'string', description: 'Billed item code, such as panel or labor.' },
          { name: 'amount', type: 'number', description: 'Amount billed for this line item, in the dataset’s currency units.' },
        ],
      },
    ],
    relations: [
      { from: 'policy', to: 'claim', fromColumn: 'policy_id', toColumn: 'policy_id', description: 'One policy has zero or more submitted claims.' },
      { from: 'claim', to: 'invoice', fromColumn: 'claim_id', toColumn: 'claim_id', description: 'One claim has zero or more supporting invoices.' },
      { from: 'invoice', to: 'line_item', fromColumn: 'invoice_id', toColumn: 'invoice_id', description: 'One invoice has zero or more billed line items.' },
    ],
  },
];
