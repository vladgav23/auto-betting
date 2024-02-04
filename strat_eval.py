import pandas as pd
import numpy as np

holdout = pd.read_csv("implementation/flumine/model/predictions/holdout-model-predictions.csv")
strat = pd.read_csv("monitoring/logs/20240202_1724_trade_strat_test.csv")

holdout['seconds_to_start_rounded'] = holdout['seconds_to_start'].apply(lambda x: round(x,2))

# Step 1: Filter to get the last_traded_price for mover=True entries
mover_true_prices = holdout[holdout['mover']].groupby(['market_id', 'seconds_to_start_rounded'])['last_traded_price'].mean().reset_index()
mover_true_prices.rename(columns={'last_traded_price': 'mover_true_last_price'}, inplace=True)

# Step 2: Join this back to the original dataframe on market_id and seconds_to_start_rounded
holdout_df_with_mover_price = pd.merge(holdout, mover_true_prices, on=['market_id', 'seconds_to_start_rounded'], how='left')

# Step 3: Calculate the difference between last_traded_price and mover_true_last_price
holdout_df_with_mover_price['price_difference'] = holdout_df_with_mover_price['last_traded_price'] - holdout_df_with_mover_price['mover_true_last_price']

# Display the dataframe to verify the results
holdout_df_with_mover_price[['market_id', 'selection_id', 'seconds_to_start_rounded', 'last_traded_price', 'mover_true_last_price', 'price_difference']].head()

holdout_df_with_mover_price['seconds_to_start'] = holdout_df_with_mover_price['seconds_to_start'].apply(lambda x: round(x,2))
strat['seconds_to_start'] = strat['order_notes'].str.split(',').str[4].astype(float).apply(lambda x: round(x, 2))
strat['predicted_wap'] = strat['order_notes'].str.split(',').str[2].astype(float).apply(lambda x: round(x, 2))

merged = strat.merge(holdout_df_with_mover_price, on=['market_id','selection_id','seconds_to_start'])


merged['profit_minus_commission'] = np.where(
    merged['profit'] < 0,
    merged['profit'],
    merged['profit'] * (1 - merged['commission']))

merged['month'] = pd.to_datetime(merged['date_time_placed'],format='mixed').dt.month


merged.query('(side=="BACK" and price > last_traded_price)').groupby(['mover','side'])['profit_minus_commission'].sum()

merged['exposure'] = merged['size_matched'] * (merged['price_matched'] - 1)



merged.query('(side=="LAY" and mover_true_last_price/last_traded_price >= 1 and predicted_wap/price >= 1.007)')['profit_minus_commission'].sum()

# Good strats
profit = merged.query('(side=="LAY" and 1.03 > (predicted_wap / last_traded_price) > 1.015)').sample(100)[
    'profit_minus_commission'].sum()

results = []
for i in range(0,10000):
    profit = merged.query('(side=="LAY" and mover_true_last_price/last_traded_price >= 1 and predicted_wap/price >= 1.007)').sample(100)['profit_minus_commission'].sum()
    results.append(profit)

results = np.array(results)

plt.boxplot(results, vert=False)
plt.show()


merged.query('(side=="LAY" and 1.03 > (predicted_wap / last_traded_price) > 1.015 and last_traded_price < 7)').groupby(['mover','side'])['exposure'].sum()


merged.query('(side=="LAY" and predicted_max_price / price > 1.13)').groupby(['mover','side'])['profit_minus_commission'].sum()
merged.query('(side=="LAY" and predicted_min_price / price_matched)').groupby(['mover','side'])['exposure'].sum()


226/3773

merged.query('seconds_to_start < 60').groupby(['mover','side'])['profit_minus_commission'].mean()