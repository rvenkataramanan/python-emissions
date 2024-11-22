import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the CSV file
file_path = r'ghg-emissions-by-sector-stacked.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Handle NaN and infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Define sector columns
sector_columns = [
    'Greenhouse gas emissions from other fuel combustion',
    'Greenhouse gas emissions from bunker fuels',
    'Greenhouse gas emissions from waste',
    'Greenhouse gas emissions from buildings',
    'Greenhouse gas emissions from industry',
    'Fugitive emissions of greenhouse gases from energy production',
    'Greenhouse gas emissions from agriculture',
    'Greenhouse gas emissions from manufacturing and construction',
    'Greenhouse gas emissions from transport',
    'Greenhouse gas emissions from electricity and heat'
]

# Initialize dictionaries and lists to store predictions and metrics
all_predictions = {}
global_predictions = []
total_rolling_mse = []

# Set window length
window_length = 5

# Loop over each sector
for sector in tqdm(sector_columns, desc="Processing sectors"):
    print(f"\nProcessing sector: {sector}")
    scaler = MinMaxScaler()
    sector_rolling_mse = []

    # Loop over each country (Entity)
    for entity in tqdm(df['Entity'].unique(), desc="Processing entities", leave=False):
        entity_data = df[df['Entity'] == entity]
        reg_entity = entity_data[sector].values.reshape(-1, 1)
        years = entity_data['Year'].values

        # Handle potential NaN or inf values
        reg_entity = np.nan_to_num(reg_entity, nan=0.0, posinf=0.0, neginf=0.0)

        reg_entity_scaled = scaler.fit_transform(reg_entity)

        if len(reg_entity_scaled) <= window_length:
            print(f"Skipping {entity} for {sector} due to insufficient data")
            continue

        X_train = []
        y_train = []
        for i in range(window_length, len(reg_entity_scaled)):
            X_train.append(reg_entity_scaled[i-window_length:i, 0])
            y_train.append(reg_entity_scaled[i, 0])
        X_train, y_train = np.array(X_train, dtype='float32'), np.array(y_train, dtype='float32')
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(window_length, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        try:
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
        except Exception as e:
            print(f"Error training model for {entity} in {sector}: {str(e)}")
            continue

        rolling_predictions = []
        rolling_mse = []
        rolling_years = years[window_length:]

        for i in range(window_length, len(reg_entity_scaled)):
            X_window = reg_entity_scaled[i-window_length:i].reshape(1, window_length, 1)
            y_window = reg_entity_scaled[i:i+1].reshape(-1, 1)
            y_pred_window = model.predict(X_window, verbose=0)
            rolling_predictions.append(y_pred_window[0, 0])
            mse_window = mean_squared_error(y_window, y_pred_window)
            rolling_mse.append(mse_window)

        rolling_predictions_scaled = np.array(rolling_predictions).reshape(-1, 1)
        rolling_predictions_original = scaler.inverse_transform(rolling_predictions_scaled)

        future_years = 10
        last_window = reg_entity_scaled[-window_length:].reshape(1, window_length, 1)
        future_predictions = []
        future_years_list = [years[-1] + i for i in range(1, future_years + 1)]

        for _ in range(future_years):
            next_pred = model.predict(last_window, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_window = np.append(last_window[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

        future_predictions_scaled = np.array(future_predictions).reshape(-1, 1)
        future_predictions_original = scaler.inverse_transform(future_predictions_scaled)

        sector_rolling_mse.append(np.mean(rolling_mse))

        predictions_df = pd.DataFrame({
            'Entity': [entity] * (len(rolling_predictions_original) + len(future_predictions_original)),
            'Year': np.concatenate([rolling_years, future_years_list]),
            'Predicted Emissions': np.concatenate([rolling_predictions_original.flatten(), future_predictions_original.flatten()])
        })

        # Store predictions for this entity and sector
        if entity not in all_predictions:
            all_predictions[entity] = {}
        all_predictions[entity][sector] = predictions_df

    total_rolling_mse.append({'Sector': sector, 'mse': np.mean(sector_rolling_mse)})

    # Calculate global predictions for this sector
    sector_predictions = pd.concat([pred for entity_preds in all_predictions.values() for pred in entity_preds.values() if pred.columns[2] == sector])
    global_predictions_for_sector = sector_predictions.groupby('Year')['Predicted Emissions'].sum().reset_index()
    global_predictions.append({
        'Sector': sector,
        'Global Predictions': global_predictions_for_sector
    })

# Create the final combined dataframe
final_df_rows = []
for entity, sector_data in all_predictions.items():
    for sector, predictions in sector_data.items():
        for _, row in predictions.iterrows():
            final_df_rows.append({
                'Entity': entity,
                'Year': row['Year'],
                sector: row['Predicted Emissions']
            })

final_df = pd.DataFrame(final_df_rows)

# Pivot the dataframe to have sectors as columns
final_df = final_df.pivot_table(index=['Entity', 'Year'], columns='variable', values='value').reset_index()

# Reorder columns to match the original dataset structure
column_order = ['Entity', 'Code', 'Year'] + sector_columns
final_df['Code'] = 'N/A'  # Add a placeholder 'Code' column
final_df = final_df.reindex(columns=column_order)

# Export the combined predictions to a CSV file
final_df.to_csv('final_sector_predictions_combined.csv', index=False)
print("Final sector predictions formatted as the original dataset exported to 'final_sector_predictions_combined.csv'.")

# Reformat global predictions to include sector-wise emissions per year
global_predictions_formatted = []
for item in global_predictions:
    sector = item['Sector']
    for _, row in item['Global Predictions'].iterrows():
        global_predictions_formatted.append({
            'Year': row['Year'],
            sector: row['Predicted Emissions']
        })

# Combine the global predictions into a single DataFrame
global_predictions_df = pd.DataFrame(global_predictions_formatted)

# Group by Year and aggregate emissions by sector
global_predictions_df = global_predictions_df.groupby(['Year'], as_index=False).sum()

# Export the global predictions to a CSV file
global_predictions_df.to_csv('global_predictions_formatted.csv', index=False)
print("Global sector predictions formatted as the original dataset exported to 'global_predictions_formatted.csv'.")

# Export total rolling MSE to a CSV file
total_rolling_mse_df = pd.DataFrame(total_rolling_mse)
total_rolling_mse_df.to_csv('total_rolling_mse.csv', index=False)
print("Total rolling MSE exported to 'total_rolling_mse.csv'.")

model.save(f'lstm_model_{sector}_{entity}.h5')
