import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

pd.options.display.max_columns = None

val_data_scored = pd.read_csv("implementation/flumine/model/predictions/holdout-model-predictions.csv")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

val_data_scored.dropna(inplace=True)

mae = mean_absolute_error(val_data_scored['target_wap'], val_data_scored['predicted_wap'])
mse = mean_squared_error(val_data_scored['target_wap'], val_data_scored['predicted_wap'])
rmse = np.sqrt(mse)
r2 = r2_score(val_data_scored['target_wap'], val_data_scored['predicted_wap'])

val_data_scored.to_csv("preds.csv")

# val_data_scored_260['best_back_price_percentile'] = pd.qcut(val_data_scored_260['best_back'], 20)
# val_data_scored_260['mean_pred_min'] = val_data_scored_260.groupby(['seconds','best_back_price_percentile'])['pred_min'].transform('mean')
# val_data_scored_260['pred_min_index'] = val_data_scored_260['pred_min'] / val_data_scored_260['mean_pred_min']
#
# val_data_scored_260['best_lay_price_percentile'] = pd.qcut(val_data_scored_260['best_lay'], 20)
# val_data_scored_260['mean_pred_max'] = val_data_scored_260.groupby(['seconds','best_lay_price_percentile'])['pred_max'].transform('mean')
# val_data_scored_260['pred_max_index'] = val_data_scored_260['pred_max'] / val_data_scored_260['mean_pred_max']
#

criteria = val_data_scored.query('pred_decrease >= 0.8 and (pred_decrease - pred_increase >= 0.1)'
                                     ).copy()

# # Sort the DataFrame
# sorted_df = criteria.sort_values(by=['market_id', 'selection_id', 'seconds_to_start'], ascending=[True, True, False])
#
# # Drop duplicates keeping the first row of each group
# grouped_df = sorted_df.drop_duplicates(subset=['market_id', 'selection_id'])

criteria.to_csv('preds_idx.csv')

def plot_market_data_corrected_x_axis(market_data, selection_id):
    market_data = market_data.sort_values(by='seconds_to_start', ascending=True)

    selection_data = market_data[market_data['selection_id'] == selection_id]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting 'prediction' and 'target' for increase and decrease on the primary axis
    ax1.plot(selection_data['seconds_to_start'], selection_data['pred_increase'], linestyle=':', color='#FF0066',
             label='Pred Increase')
    ax1.plot(selection_data['seconds_to_start'], selection_data['actual_increase'], linestyle='--', color='#FF0066',
             label='Actual Increase')
    ax1.plot(selection_data['seconds_to_start'], selection_data['pred_decrease'], linestyle=':', color='#00AEEF',
             label='Pred Decrease')
    ax1.plot(selection_data['seconds_to_start'], selection_data['actual_decrease'], linestyle='--', color='#00AEEF',
             label='Actual Decrease')

    ax1.set_xlabel('Seconds to Start')
    ax1.set_ylabel('Values')
    ax1.set_title(f"Market ID: {market_data['market_id'].iloc[0]} Selection ID: {selection_id}")
    ax1.invert_xaxis()  # Inverting x-axis
    ax1.legend(loc='upper left')

    # Creating a second y-axis for 'best_back' and 'best_lay'
    ax2 = ax1.twinx()
    ax2.plot(selection_data['seconds_to_start'], selection_data['best_back'], linestyle='-', color='#00AEEF',
             label='Best Back')
    ax2.plot(selection_data['seconds_to_start'], selection_data['best_lay'], linestyle='-', color='#FF0066',
             label='Best Lay')
    ax2.set_ylabel('Best Values')
    ax2.legend(loc='upper right')

    plt.show()

# Plot the data using the corrected function with the filtered data

unique_markets = val_data_scored['market_id'].unique()

market_data = val_data_scored[
    val_data_scored['market_id'] == 1.216629231]
selections = market_data.sort_values('best_back')['selection_id'].unique()
for sel in selections:
    plot_market_data_corrected_x_axis(market_data, sel)


import numpy as np
import matplotlib.pyplot as plt

# Calculate the percentiles based on predicted values
val_data_scored['Percentile_increase'] = pd.qcut(val_data_scored['pred_increase'], q=100, labels=False, duplicates='drop')

val_data_scored['Percentile_decrease'] = pd.qcut(val_data_scored['pred_decrease'], q=100, labels=False, duplicates='drop')


val_data_scored['pct_best_back'] = pd.qcut(val_data_scored['best_back'], q=20, labels=False, duplicates='drop')
val_data_scored['pct_best_lay'] = pd.qcut(val_data_scored['best_lay'], q=20, labels=False, duplicates='drop')



# Group by percentile and calculate average predicted and actual values
grouped_data = val_data_scored.groupby('pct_best_lay').agg(Avg_Lay=('best_lay','mean'),
                                                           Avg_Predicted=('pred_increase', 'mean'),
                                                                  Avg_Actual=('actual_increase', 'mean')).reset_index()

grouped_data_dec = val_data_scored.groupby('pct_best_back').agg(Avg_Back=('best_back','mean'),
    Avg_Predicted=('pred_decrease', 'mean'),
                                                                  Avg_Actual=('actual_decrease', 'mean')).reset_index()

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Chart for Increase
axs[0].plot(grouped_data['Avg_Lay'], grouped_data['Avg_Predicted'], label='Average Predicted', marker='o')
axs[0].plot(grouped_data['Avg_Lay'], grouped_data['Avg_Actual'], label='Average Actual', marker='x')
axs[0].set_title('Average Predicted vs. Actual for Each Percentile (Increase)')
axs[0].set_xlabel('Percentile')
axs[0].set_ylabel('Value')
axs[0].legend()

# Chart for Decrease
axs[1].plot(grouped_data_dec['Avg_Back'], grouped_data_dec['Avg_Predicted'], label='Average Predicted', marker='o')
axs[1].plot(grouped_data_dec['Avg_Back'], grouped_data_dec['Avg_Actual'], label='Average Actual', marker='x')
axs[1].set_title('Average Predicted vs. Actual for Each Percentile (Decrease)')
axs[1].set_xlabel('Percentile')
axs[1].set_ylabel('Value')
axs[1].legend()

plt.tight_layout()
plt.show()

