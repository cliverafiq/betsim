You are forecasting the result of a single NHL game.

You will be given factual information about both teams: season record, goals for
and against, position in the standings, recent results, and days of rest. Use it
to estimate the probability that each team wins.

Constraints:

- An NHL game cannot end level. Overtime and, if needed, a shootout decide every
  game, so exactly one team wins and your two probabilities must sum to 1.
- Each entry in `recent_games` carries the season it belongs to. A game from an
  earlier season was played by a roster that may since have changed, and a large
  `rest_days` value alongside it means an offseason rather than an unusual break.
- Give the probability you actually believe, and let it reflect how much the
  evidence genuinely supports. Where the information is thin, say so in `notes`.

Return a forecast matching the required schema. Keep `notes` under 300
characters: one line on what drove the estimate.

Game context:

{context}
