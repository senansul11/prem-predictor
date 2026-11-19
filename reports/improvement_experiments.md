# Model improvement experiments

Four ideas were tested on the Dixon-Coles goals model, one at a time, with the usual rules:
tune on validation seasons 2016-21 only, then score on test seasons 2021-27 (walk-forward,
weekly refits). RPS: lower is better.

## Result
| period | model | n | rps | log_loss | brier | accuracy |
|---|---|---|---|---|---|---|
| validation 2016-21 | Baseline Dixon-Coles (v1) | 1900 | 0.1968 | 0.9565 | 0.5660 | 0.5542 |
| validation 2016-21 | + shot-based xG proxy only | 1900 | 0.1961 | 0.9548 | 0.5649 | 0.5547 |
| validation 2016-21 | + team home advantage only (ridge 10) | 1900 | 0.1967 | 0.9563 | 0.5658 | 0.5547 |
| validation 2016-21 | + xG proxy and team home (ridge 10) | 1900 | 0.1960 | 0.9546 | 0.5647 | 0.5532 |
| validation 2016-21 | FINAL: xG proxy + retuned settings | 1900 | 0.1959 | 0.9544 | 0.5645 | 0.5532 |
| test 2021-27 | Baseline Dixon-Coles (v1) | 2000 | 0.2012 | 0.9797 | 0.5832 | 0.5295 |
| test 2021-27 | + shot-based xG proxy only | 2000 | 0.2009 | 0.9788 | 0.5823 | 0.5340 |
| test 2021-27 | + team home advantage only (ridge 10) | 2000 | 0.2016 | 0.9805 | 0.5837 | 0.5280 |
| test 2021-27 | + xG proxy and team home (ridge 10) | 2000 | 0.2011 | 0.9792 | 0.5826 | 0.5325 |
| test 2021-27 | FINAL: xG proxy + retuned settings | 2000 | 0.2001 | 0.9766 | 0.5808 | 0.5365 |

RPS improvement on test (baseline minus final): **+0.0012**.
Average error in predicted total goals per match (model - actual): baseline -0.048, final -0.099.

Final settings: `{"xi": 0.003, "season_decay": 0.8, "ridge": 0.05, "w_e1": 0.5, "red_w": 1.0, "use_momentum": false, "xg_blend": 0.5, "team_home_ridge": 0.0, "max_goals": 10}`

## Ideas 2 and 3 (draws, goal level): no change made
The diagnosis showed draw probabilities and the goal level were already well calibrated
(draws: 27.6% predicted vs 26.3% actual in the closest games). Post-hoc corrections made test RPS
slightly *worse* (baseline 0.20124): outcome recalibration 0.20155, draw scaling 0.20132,
trailing goal-level rescaling 0.20173-0.20211. The one big miss is 2023-24, an unusually
high-scoring season (3.28 goals/game), which no stable model predicts.

## Idea 1: shot-based xG proxy
Goals ~ 0.333 x shots on target + -0.013 x other shots (fitted on seasons up to 2020-21).
Each simulated "target" mixes real goals with this proxy; chosen mix: **50% proxy**.

## Idea 4: team-specific home advantage (current clubs)
Learned from historic results (fitted with ridge 10 for display), shrunk towards the league average. **Not used in the final model**: it did not beat the plain model on validation by the required margin, and was slightly worse on test.
`home_goal_boost_%` is the extra boost to the team's home goal rate vs the average club.
| team | home_goal_boost_% | home_ppg | away_ppg | home_minus_away_ppg | home_matches_in_history |
|---|---|---|---|---|---|
| Manchester United | 7.740 | 1.908 | 1.528 | 0.379 | 195 |
| Fulham | 5.468 | 1.303 | 0.958 | 0.345 | 119 |
| Arsenal | 3.714 | 2.231 | 1.631 | 0.600 | 195 |
| Brentford | 3.053 | 1.560 | 1.110 | 0.450 | 100 |
| Brighton & Hove Albion | 2.132 | 1.472 | 1.114 | 0.358 | 176 |
| Coventry City | 2.105 | 1.200 | 0.600 | 0.600 | 5 |
| Newcastle United | 1.948 | 1.625 | 1.131 | 0.494 | 176 |
| Leeds United | 1.806 | 1.358 | 0.975 | 0.383 | 81 |
| Tottenham Hotspur | 1.212 | 1.872 | 1.426 | 0.446 | 195 |
| Aston Villa | 0.597 | 1.688 | 1.268 | 0.420 | 138 |
| Manchester City | 0.227 | 2.446 | 2.097 | 0.349 | 195 |
| Hull City | 0.151 | 1.458 | 0.417 | 1.042 | 24 |
| Everton | 0.110 | 1.533 | 1.072 | 0.462 | 195 |
| Chelsea | -0.294 | 1.908 | 1.610 | 0.297 | 195 |
| Crystal Palace | -0.486 | 1.262 | 1.138 | 0.123 | 195 |
| Ipswich Town | -5.066 | 0.583 | 0.792 | -0.208 | 24 |
| AFC Bournemouth | -5.444 | 1.420 | 1.025 | 0.395 | 157 |
| Sunderland | -6.134 | 1.256 | 0.744 | 0.512 | 43 |
| Liverpool | -7.589 | 2.323 | 1.877 | 0.446 | 195 |
| Nottingham Forest | -11.275 | 1.309 | 1.062 | 0.247 | 81 |

## Tuning log (validation RPS)
| param | value | val_rps |
|---|---|---|
| xg_blend | 0.0 (current) | 0.19760 |
| xg_blend | 0.25 | 0.19684 |
| xg_blend | 0.5 | 0.19672 |
| xg_blend | 0.75 | 0.19730 |
| xg_blend | 1.0 | 0.19864 |
| team_home_ridge | 0.0 (current) | 0.19672 |
| team_home_ridge | 100.0 | 0.19669 |
| team_home_ridge | 50.0 | 0.19668 |
| team_home_ridge | 25.0 | 0.19665 |
| team_home_ridge | 10.0 | 0.19663 |
| team_home_ridge | 5.0 | 0.19665 |
| xi | 0.002 (current) | 0.19672 |
| xi | 0.001 | 0.19728 |
| xi | 0.0015 | 0.19693 |
| xi | 0.003 | 0.19661 |
| ridge | 0.05 (current) | 0.19661 |
| ridge | 0.15 | 0.19677 |
| ridge | 0.3 | 0.19701 |
