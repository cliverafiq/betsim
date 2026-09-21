You are managing a hypothetical sports-betting bankroll. No real money is
involved at any point.

Earlier, before seeing any prices, you estimated a win probability for each of
today's games. Those estimates are shown below as `p_blind`, alongside the price
now available. Decide which bets to place today, if any.

The system enforces these rules. A bet that breaks one is **rejected, not
adjusted**, and the rejection is recorded:

- Stakes are in units, minimum 1 unit.
- Each selection's tier is assigned from its price, and caps the stake for that
  bet. The cap in units is given per selection as `tier_cap_units`.
- Your total stake across unsettled bets may not exceed
  `remaining_exposure_units`.
- At most one bet per game, and only on games that have not started.
- Bets are processed in the order you return them. If the exposure cap binds,
  later bets are rejected, so put the ones you want most first.

**Placing no bets at all is a perfectly good outcome.** There is no minimum
number of bets, no quota to meet, and no penalty for sitting out a day. A price
is only worth taking if you think it is wrong.

For every bet give `p_revised`: your probability for that selection now that you
have seen the price. It may equal your earlier estimate or differ from it.
Keep each `reason` under 300 characters and `day_notes` under 500.

Your position and today's slate:

{slate}
