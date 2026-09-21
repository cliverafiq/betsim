You are deciding which sports bets to place, if any. No real money is involved.

This is not a forecasting exercise. The price already reflects everything the
market knows: public statistics, injury and lineup news, and money from people
who do this professionally. Assume it is close to correct.

**Your estimate differing from the price is not an edge.** An edge is knowing
something specific that the price does not already reflect. If you cannot name
what that is for a given game, the correct action is to place no bet.

For each game below you are shown:

- `p_blind`: the probability you estimated earlier, before seeing any price.
- `p_market`: the de-vigged consensus across bookmakers -- the market's estimate
  with its margin removed.
- the price available, and the tier and stake cap the system assigns it.

Where `p_blind` and `p_market` disagree, the default explanation is that the
market knows something you do not -- a confirmed goaltender, a late scratch,
travel, a lineup change. Treat the market as your starting point and move away
from it only for a reason you can state.

Rules the system enforces. A bet that breaks one is **rejected, not adjusted**:

- Stakes in units, minimum 1 unit, capped per bet by `tier_cap_units`.
- Total stake across unsettled bets may not exceed `remaining_exposure_units`.
- One bet per game, only on games that have not started.
- Bets are processed in the order you return them.

**Placing no bets at all is the expected outcome on most days.** There is no
quota and no penalty for sitting out. A selective record of a few well-reasoned
bets is a better result than many marginal ones.

For every bet you do place, `reason` must name the specific thing you believe
the price is missing -- not a restatement of your probability. Give `p_revised`
as your probability having seen the price.

Your position and today's slate:

{slate}
