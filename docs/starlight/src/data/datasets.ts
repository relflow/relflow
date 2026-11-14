import type { Dataset } from '../datasets/types';

/** Illustrative source tables; relflow also accepts these records already nested. */
export const datasets: Dataset[] = [
  {
    id: 'tickets',
    label: 'Support tickets',
    observation: 'One support ticket at triage.',
    description: 'The ticket supplies product and priority; its messages supply the conversation available at triage. Historical outcomes teach escalation and time to resolution.',
    entities: [
      {
        id: 'ticket', name: 'Ticket',
        description: 'The observation represented by TicketModel, with historical target labels alongside the known ticket fields.',
        columns: [
          { name: 'ticket_id', type: 'string', role: 'key', description: 'Illustrative source key joining a ticket to its messages. This key is not an input field in the example model.' },
          { name: 'product', type: 'string', description: 'Product associated with the support request, such as analytics. Modeled as a Category.' },
          { name: 'priority', type: 'string', description: 'Priority known at triage: low, normal, or high. Modeled with a fixed Enum vocabulary.' },
          { name: 'escalated', type: 'boolean', role: 'target', description: 'Recorded escalation outcome for a historical ticket. The model learns to predict this Boolean; it is absent from prediction inputs.' },
          { name: 'hours_to_resolve', type: 'number', role: 'target', description: 'Recorded time to resolution, in hours. This Number target is learned jointly with escalation and is absent from prediction inputs.' },
        ],
      },
      {
        id: 'message', name: 'Message',
        description: 'One conversation message, placed in the ticket’s messages collection in conversation order.',
        columns: [
          { name: 'message_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one message. The example model uses the message fields rather than this key.' },
          { name: 'ticket_id', type: 'string', role: 'reference', description: 'Illustrative join key linking this message to its ticket. After grouping, the message is an item in that ticket’s messages collection.' },
          { name: 'author', type: 'string', description: 'Author role for this message, such as customer or agent. Modeled as a Category.' },
          { name: 'body', type: 'string', description: 'Message text available at triage. Modeled as Text so the conversation can be supplied without hand-written text features.' },
          { name: 'minutes_since_previous', type: 'number', description: 'Minutes since the preceding message in the supplied conversation. This elapsed interval is modeled as a Number.' },
        ],
      },
    ],
    relations: [
      { from: 'ticket', to: 'message', fromColumn: 'ticket_id', toColumn: 'ticket_id', description: 'One ticket has zero or more messages available at triage, grouped into messages.' },
    ],
  },
  {
    id: 'churn',
    label: 'Customer churn',
    observation: 'One subscriber at a prediction cutoff.',
    description: 'A subscriber has two separate histories: monthly usage and support tickets. Their known context is used to predict cancellation within the next 90 days.',
    entities: [
      {
        id: 'subscriber', name: 'Subscriber',
        description: 'The subscriber observation represented by ChurnModel, with both histories assembled as child collections.',
        columns: [
          { name: 'subscriber_id', type: 'string', role: 'key', description: 'Illustrative source key joining a subscriber to both histories. This key is not a modeled input in the example.' },
          { name: 'plan', type: 'string', description: 'Subscription plan known at the cutoff, such as pro. Modeled as a Category.' },
          { name: 'tenure_months', type: 'number', description: 'Subscriber tenure in months at the cutoff. Modeled as a Number.' },
          { name: 'churn_90d', type: 'boolean', role: 'target', description: 'Whether the subscriber canceled within 90 days after the cutoff. This historical Boolean label is absent from prediction inputs.' },
        ],
      },
      {
        id: 'monthly_usage', name: 'Monthly usage',
        description: 'One observed month in the subscriber’s monthly_usage collection, ordered from older to newer months.',
        columns: [
          { name: 'usage_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one monthly usage record. It is not declared in the example model.' },
          { name: 'subscriber_id', type: 'string', role: 'reference', description: 'Illustrative source key identifying the subscriber whose monthly_usage collection receives this record.' },
          { name: 'month', type: 'date', description: 'Date identifying the observed month. Period("month_of_year") encodes its calendar month, rather than elapsed recency.' },
          { name: 'minutes', type: 'number', description: 'Usage minutes recorded for this month. Modeled as a Number.' },
          { name: 'data_gb', type: 'number', description: 'Data usage in gigabytes for this month. Modeled as a Number.' },
          { name: 'overage_fee', type: 'number', description: 'Overage charge recorded for this month, in the dataset’s currency units. Modeled as a Number.' },
        ],
      },
      {
        id: 'support_ticket', name: 'Support ticket',
        description: 'One ticket in the subscriber’s support_tickets history, using only information available at the cutoff.',
        columns: [
          { name: 'ticket_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one historical support ticket. It is not declared in the example model.' },
          { name: 'subscriber_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this ticket in its subscriber’s support_tickets collection.' },
          { name: 'channel', type: 'string', description: 'Support channel used for this ticket, such as chat. Modeled as a Category.' },
          { name: 'resolved', type: 'boolean', description: 'Resolution status known at the subscriber cutoff. This Boolean is a known input here, rather than the future churn target.' },
          { name: 'days_open', type: 'number', description: 'Recorded time open, in days, available at the cutoff. Modeled as a Number.' },
        ],
      },
    ],
    relations: [
      { from: 'subscriber', to: 'monthly_usage', fromColumn: 'subscriber_id', toColumn: 'subscriber_id', description: 'One subscriber has zero or more monthly usage records, grouped into monthly_usage.' },
      { from: 'subscriber', to: 'support_ticket', fromColumn: 'subscriber_id', toColumn: 'subscriber_id', description: 'One subscriber has zero or more historical tickets, grouped into support_tickets.' },
    ],
  },
  {
    id: 'demand',
    label: 'Demand forecasting',
    observation: 'One product at one store, with supplied future weeks.',
    description: 'Past sales and planned future weeks are separate collections. The model predicts units_sold for each supplied horizon record using the store, product, sales history, and known promotions.',
    entities: [
      {
        id: 'store_product', name: 'Store product',
        description: 'The store-and-product observation represented by DemandModel, with history and horizon collections.',
        columns: [
          { name: 'store_product_id', type: 'string', role: 'key', description: 'Illustrative source key for a store-and-product pair. This join key is not declared in the model; store and sku are the modeled identifiers.' },
          { name: 'store', type: 'string', description: 'Store identifier, such as store-4. Modeled as a Category.' },
          { name: 'sku', type: 'string', description: 'Product identifier, such as boots-9. Modeled as a Category.' },
        ],
      },
      {
        id: 'sales_week', name: 'Sales week',
        description: 'One observed past week in history, ordered from older to newer weeks.',
        columns: [
          { name: 'sales_week_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one observed weekly sales record. It is not a modeled input.' },
          { name: 'store_product_id', type: 'string', role: 'reference', description: 'Illustrative source key placing this weekly record in the matching store-and-product history.' },
          { name: 'week', type: 'date', description: 'Date identifying a past sales week. Period("week_of_year") encodes its calendar week.' },
          { name: 'units_sold', type: 'number', description: 'Observed units sold during this past week. This value is a known input; units_sold in the forecast horizon is the prediction target.' },
          { name: 'promotion', type: 'boolean', description: 'Whether a promotion applied during this past week. Modeled as a Boolean input.' },
          { name: 'stockout', type: 'boolean', description: 'Whether a stockout was recorded during this past week. Modeled as a Boolean input.' },
        ],
      },
      {
        id: 'forecast_week', name: 'Forecast week',
        description: 'One supplied future week in horizon. The model returns a units_sold prediction aligned with each of these records.',
        columns: [
          { name: 'forecast_week_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one requested future week. The caller supplies the horizon records; the model does not generate their number or dates.' },
          { name: 'store_product_id', type: 'string', role: 'reference', description: 'Illustrative source key placing this future week in the matching store-and-product horizon.' },
          { name: 'week', type: 'date', description: 'Known date for the future week being forecast. Period("week_of_year") encodes its calendar week.' },
          { name: 'promotion', type: 'boolean', description: 'Promotion plan known for this future week when making the forecast. Modeled as a Boolean input.' },
          { name: 'units_sold', type: 'number', role: 'target', description: 'Units eventually sold in this supplied future week. Present as a label in historical training observations and absent from prediction inputs.' },
        ],
      },
    ],
    relations: [
      { from: 'store_product', to: 'sales_week', fromColumn: 'store_product_id', toColumn: 'store_product_id', description: 'One store-and-product pair has zero or more observed weeks, grouped into history.' },
      { from: 'store_product', to: 'forecast_week', fromColumn: 'store_product_id', toColumn: 'store_product_id', description: 'One store-and-product pair has supplied future weeks, grouped into horizon for per-week predictions.' },
    ],
  },
  {
    id: 'credit',
    label: 'Credit risk',
    observation: 'One credit application at assessment.',
    description: 'The application combines its known terms with existing credit accounts and recent inquiries. A historical default label teaches the model to estimate default within 12 months.',
    entities: [
      {
        id: 'application', name: 'Application',
        description: 'The application observation represented by CreditModel, with account and inquiry snapshots assembled at assessment.',
        columns: [
          { name: 'application_id', type: 'string', role: 'key', description: 'Illustrative key for an application snapshot and its related records. This join key is not a modeled input.' },
          { name: 'product', type: 'string', description: 'Requested credit product, such as installment. Modeled as a Category.' },
          { name: 'requested_amount', type: 'number', description: 'Amount requested in the application, in the dataset’s currency units. Modeled as a Number.' },
          { name: 'default_12m', type: 'boolean', role: 'target', description: 'Whether default occurred within 12 months after assessment. This historical Boolean label is absent from prediction inputs.' },
        ],
      },
      {
        id: 'account', name: 'Credit account',
        description: 'One existing account in the application’s accounts snapshot. This diagram depicts the supplied snapshot, rather than a complete credit-bureau database.',
        columns: [
          { name: 'account_id', type: 'string', role: 'key', description: 'Illustrative source key for one account record in this application snapshot. It is not declared in the example model.' },
          { name: 'application_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this account snapshot in the assessed application’s accounts collection.' },
          { name: 'account_type', type: 'string', description: 'Type of existing credit account, such as card. Modeled as a Category.' },
          { name: 'balance', type: 'number', description: 'Account balance known at assessment, in the dataset’s currency units. Modeled as a Number.' },
          { name: 'months_open', type: 'number', description: 'Age of this account in months at assessment. Modeled as a Number.' },
          { name: 'missed_payments', type: 'number', description: 'Recorded missed-payment count available at assessment. Modeled as a Number.' },
        ],
      },
      {
        id: 'inquiry', name: 'Credit inquiry',
        description: 'One recent inquiry in the application’s inquiries snapshot, ordered from older to newer records.',
        columns: [
          { name: 'inquiry_id', type: 'string', role: 'key', description: 'Illustrative source key for one inquiry in this application snapshot. It is not declared in the example model.' },
          { name: 'application_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this inquiry in the assessed application’s inquiries collection.' },
          { name: 'lender', type: 'string', description: 'Lender associated with the inquiry. Modeled as a Category.' },
          { name: 'age_days', type: 'number', description: 'Days elapsed between the inquiry and assessment. This recency value is modeled as a Number.' },
        ],
      },
    ],
    relations: [
      { from: 'application', to: 'account', fromColumn: 'application_id', toColumn: 'application_id', description: 'One application snapshot contains zero or more existing account records, grouped into accounts.' },
      { from: 'application', to: 'inquiry', fromColumn: 'application_id', toColumn: 'application_id', description: 'One application snapshot contains zero or more recent inquiries, grouped into inquiries.' },
    ],
  },
  {
    id: 'maintenance',
    label: 'Predictive maintenance',
    observation: 'One asset at a prediction cutoff.',
    description: 'An asset has sensor observations and completed maintenance work. The model uses those two histories to predict failure within the next 30 days.',
    entities: [
      {
        id: 'asset', name: 'Asset',
        description: 'The asset observation represented by MaintenanceModel, with both histories assembled at the cutoff.',
        columns: [
          { name: 'asset_id', type: 'string', role: 'key', description: 'Illustrative source key joining an asset to its sensor and maintenance records. This key is not a modeled input.' },
          { name: 'asset_type', type: 'string', description: 'Type of equipment, such as pump. Modeled as a Category.' },
          { name: 'age_years', type: 'number', description: 'Asset age in years at the cutoff. Modeled as a Number.' },
          { name: 'failure_30d', type: 'boolean', role: 'target', description: 'Whether the asset failed within 30 days after the cutoff. This historical Boolean label is absent from prediction inputs.' },
        ],
      },
      {
        id: 'sensor_reading', name: 'Sensor reading',
        description: 'One observation in sensor_readings, ordered from older to newer records.',
        columns: [
          { name: 'reading_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one sensor observation. It is not declared in the example model.' },
          { name: 'asset_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this reading in its asset’s sensor_readings collection.' },
          { name: 'hour', type: 'datetime', description: 'Timestamp of this observation. Period("hour_of_day") encodes its calendar hour, rather than elapsed recency.' },
          { name: 'vibration', type: 'number', description: 'Vibration measurement in the source dataset’s units. Modeled as a Number; the example does not prescribe a physical unit.' },
          { name: 'temperature', type: 'number', description: 'Temperature measurement in the source dataset’s units. Modeled as a Number.' },
          { name: 'load', type: 'number', description: 'Equipment load recorded with this observation, using the source dataset’s scale. Modeled as a Number.' },
        ],
      },
      {
        id: 'work_order', name: 'Work order',
        description: 'One completed maintenance job in work_orders, using only work known at the cutoff.',
        columns: [
          { name: 'work_order_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one completed maintenance job. It is not declared in the example model.' },
          { name: 'asset_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this job in its asset’s work_orders collection.' },
          { name: 'type', type: 'string', description: 'Type of completed maintenance work, such as inspection. Modeled as a Category.' },
          { name: 'parts_replaced', type: 'string[]', description: 'Part labels replaced during this job. Modeled as a Set; an empty list means no parts were replaced.' },
          { name: 'age_days', type: 'number', description: 'Age of the completed job in days at the cutoff. This recency value is modeled as a Number.' },
        ],
      },
    ],
    relations: [
      { from: 'asset', to: 'sensor_reading', fromColumn: 'asset_id', toColumn: 'asset_id', description: 'One asset has zero or more sensor observations, grouped into sensor_readings.' },
      { from: 'asset', to: 'work_order', fromColumn: 'asset_id', toColumn: 'asset_id', description: 'One asset has zero or more completed maintenance jobs, grouped into work_orders.' },
    ],
  },
  {
    id: 'claims',
    label: 'Insurance claims',
    observation: 'One policy with submitted claims at review.',
    description: 'A policy contains claims, each claim has invoices, and each invoice has billed items. The model predicts both paid_amount and is_fraudulent for each supplied claim.',
    entities: [
      {
        id: 'policy', name: 'Policy',
        description: 'The policy observation represented by ClaimsModel. Its claims collection contains the submitted claims under review.',
        columns: [
          { name: 'policy_id', type: 'string', role: 'key', description: 'Illustrative source key joining a policy to its submitted claims. This key is not declared in the example model.' },
          { name: 'policy_type', type: 'string', description: 'Type of insurance policy, such as auto. Modeled as a Category.' },
          { name: 'coverage_limit', type: 'number', description: 'Policy coverage limit known at review, in the dataset’s currency units. Modeled as a Number.' },
        ],
      },
      {
        id: 'claim', name: 'Claim',
        description: 'One submitted claim in claims. Both prediction targets belong to this record, rather than to the policy as a whole.',
        columns: [
          { name: 'claim_id', type: 'string', role: 'key', description: 'Illustrative source key joining a claim to its supporting invoices. It is not declared in the example model.' },
          { name: 'policy_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this claim in the policy’s claims collection.' },
          { name: 'cause', type: 'string', description: 'Reported cause of this claim, such as collision. Modeled as a Category.' },
          { name: 'reported_after_days', type: 'number', description: 'Days between the incident and the claim being reported. Modeled as a Number.' },
          { name: 'paid_amount', type: 'number', role: 'target', description: 'Final amount paid for this claim, in the dataset’s currency units. This historical Number label is absent from prediction inputs.' },
          { name: 'is_fraudulent', type: 'boolean', role: 'target', description: 'Recorded fraud outcome for this claim. A historical Boolean label trains its fraud prediction; the label is absent from prediction inputs.' },
        ],
      },
      {
        id: 'invoice', name: 'Invoice',
        description: 'One supporting invoice in a claim’s invoices collection, with its own line_items collection.',
        columns: [
          { name: 'invoice_id', type: 'string', role: 'key', description: 'Illustrative source key joining an invoice to its billed items. It is not declared in the example model.' },
          { name: 'claim_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this invoice in the claim’s invoices collection.' },
          { name: 'provider', type: 'string', description: 'Provider identifier on this invoice. Modeled as a Hash so equality can be represented without a persistent learned vocabulary.' },
        ],
      },
      {
        id: 'line_item', name: 'Line item',
        description: 'One billed item in an invoice’s line_items collection, preserving which invoice and claim supplied it.',
        columns: [
          { name: 'line_item_id', type: 'string', role: 'key', description: 'Illustrative source key identifying one billed item. It is not declared in the example model.' },
          { name: 'invoice_id', type: 'string', role: 'reference', description: 'Illustrative join key placing this billed item in the invoice’s line_items collection.' },
          { name: 'code', type: 'string', description: 'Billed item code, such as panel or labor. Modeled as a Category.' },
          { name: 'amount', type: 'number', description: 'Amount billed for this item, in the dataset’s currency units. This is a known Number input, distinct from the claim’s final paid_amount target.' },
        ],
      },
    ],
    relations: [
      { from: 'policy', to: 'claim', fromColumn: 'policy_id', toColumn: 'policy_id', description: 'One policy has submitted claims, grouped into claims for per-claim predictions.' },
      { from: 'claim', to: 'invoice', fromColumn: 'claim_id', toColumn: 'claim_id', description: 'One claim has zero or more supporting invoices, grouped into invoices.' },
      { from: 'invoice', to: 'line_item', fromColumn: 'invoice_id', toColumn: 'invoice_id', description: 'One invoice has zero or more billed items, grouped into line_items.' },
    ],
  },
];
