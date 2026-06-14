# Jano Use Cases Beyond Supervised ML

## Scope

Jano can simulate systems whose state is calibrated from sequential observations
and needs a policy to decide when and how to update that state. The core question
is always the same:

> Given a system that degrades over time or over accumulated events, what is the
> optimal update cadence, how much history should inform each update, and what is
> the cost of not updating?

ML is the first domain, not the only one. The `UpdateableSystem` protocol
(`update` / `evaluate`) and `TemporalSystemRunner` make this explicit: Jano does
not need to know what happens inside the system. It owns the temporal geometry
and the structured reporting, while the user provides the update policy and the
system-specific update/evaluation behavior.

The sequential axis does not have to be calendar time. It can be a transaction
counter, an episode number, a batch ID or any column that defines a causal
before/after ordering.

## Use cases

### 1. Dynamic pricing with Bayesian inference

A pricing engine estimates willingness-to-pay through a Bayesian posterior.
Each batch of transactions provides evidence to update the prior into a
posterior, and the posterior determines the optimal price.

- `update`: recompute the posterior distribution from recent transactions.
- `state`: posterior parameters and the derived optimal price.
- `evaluate`: measure revenue, conversion rate and regret vs an oracle price.

Jano questions:

- How many transactions should feed each posterior update?
- Does updating every 50 transactions outperform every 500?
- How long does a posterior remain useful before the market shifts?
- What is the optimal replay buffer size (train history) for estimation quality?

This case is especially interesting because updating too frequently produces a
noisy posterior (volatile pricing), while updating too rarely produces a stale
posterior (suboptimal pricing). The optimal cadence depends on market dynamics
and can only be found by simulation.

### 2. LLM fine-tuning refresh

An organization fine-tunes a base model on its internal data (support tickets,
documentation, domain corpora). That data has temporal structure.

- `update`: run a fine-tuning job with a temporal window of data.
- `state`: the fine-tuned model checkpoint.
- `evaluate`: measure task-specific accuracy, hallucination rate, groundedness.

Jano questions:

- How often should we re-fine-tune?
- Does fine-tuning with the last 3 months of data outperform 12 months?
- How long does a fine-tuned checkpoint remain effective?

### 3. RAG knowledge base refresh

A retrieval-augmented generation system has a document index that ages.

- `update`: reindex documents from the current temporal window.
- `state`: the vector index or retrieval configuration.
- `evaluate`: measure retrieval precision, answer accuracy, groundedness.

Jano questions:

- How often should we reindex?
- Does including older documents help or hurt retrieval quality?
- When does the index become stale enough to justify the reindexing cost?

### 4. Prompt engineering stability

A prompt designed for a specific model and user population may degrade as the
model provider updates the model, the user population shifts, or the domain
evolves.

- `update`: select or revise the prompt set based on recent evaluation data.
- `state`: the current prompt configuration.
- `evaluate`: measure response quality, user satisfaction, cost per query.

Jano questions:

- How stable is the current prompt over time?
- Does the prompt need revision every week, every month, or only after a
  provider model update?

### 5. Business rule calibration

Hard-coded rules ("escalate if complaints > 3", "flag if amount > 10000")
were calibrated with historical data. The operating environment changes.

- `update`: recalibrate thresholds from recent operational data.
- `state`: the current rule parameters.
- `evaluate`: measure false positive rate, missed escalations, customer churn.

Jano questions:

- Do rules calibrated with the last 1000 events outperform those calibrated
  with 5000?
- How often should thresholds be reviewed?
- How long do current thresholds remain effective?

### 6. Portfolio rebalancing and asset allocation

A portfolio is rebalanced periodically based on estimated covariance matrices
or factor exposures.

- `update`: reestimate the covariance matrix or factor model from recent returns.
- `state`: portfolio weights.
- `evaluate`: measure Sharpe ratio, drawdown, tracking error.

Jano questions:

- Does rebalancing every 20 trading days outperform every 60?
- How much return history should inform the covariance estimate?
- Does the optimal rebalancing frequency change across volatility regimes?

### 7. Recommendation system refresh

Collaborative filtering or embedding-based recommendations age as user
preferences shift.

- `update`: recompute user/item embeddings from recent interactions.
- `state`: the recommendation model or embedding table.
- `evaluate`: measure CTR, watch-through rate, diversity.

Jano questions:

- How many user interactions are needed before refreshing recommendations?
- Does a user who has been inactive for 30 days still match their old profile?
- What is the optimal refresh cadence per user segment?

### 8. Reinforcement learning policy maintenance

An RL agent trained for a production environment (ad bidding, industrial
control, trading) faces non-stationary dynamics. Jano does not replace the RL
training loop (Gymnasium, Stable-Baselines, RLlib) but answers the
meta-question: once the agent is trained, how do you maintain it?

- `update`: retrain or fine-tune the policy with recent episodes/transitions.
- `state`: the current policy checkpoint.
- `evaluate`: off-policy evaluation of the frozen policy on new episodes.

Jano questions:

- How often should the production policy be retrained?
- How large should the replay buffer be?
- How long does a policy survive a regime change in the environment?

### 9. Alerting and operational thresholds

Monitoring systems use thresholds ("alert if CPU > 85% for 5 min") that were
calibrated under different infrastructure conditions.

- `update`: recalibrate thresholds from recent operational telemetry.
- `state`: current threshold configuration.
- `evaluate`: measure false alarm rate, missed incidents.

Jano questions:

- Do thresholds calibrated with the last 1000 alerts outperform those from
  the last 10000?
- How often should thresholds be reviewed as infrastructure evolves?

### 10. Data quality rule learning

A data pipeline has validation rules learned from historical schema behavior.
As upstream sources change, the rules may become stale.

- `update`: relearn validation rules from recent data batches.
- `state`: current rule set.
- `evaluate`: measure true anomaly detection rate vs false alarm rate.

Jano questions:

- Do rules learned from the last month of data catch more real anomalies than
  rules from the full history?
- How quickly do validation rules become obsolete after an upstream change?

## Common pattern

Every case above shares the same structure:

1. A **state** calibrated from sequential observations.
2. A **degradation** dynamic: the state becomes less effective over time or events.
3. An **update cost**: refreshing the state is not free.
4. A **policy question**: when to update, with how much history, at what cost.

Jano answers these questions through temporal simulation without embedding
domain-specific logic. The user provides `update` and `evaluate`; Jano provides
geometry, policy orchestration and structured reporting.
