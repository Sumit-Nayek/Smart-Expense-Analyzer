import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: better styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ------------------------------
# 1. Load the CSV file
# ------------------------------
csv_file = "PhonePe_Statement_Jan2026_Feb2026_transactions.csv"  # <-- change to your file
df = pd.read_csv(csv_file)

# Display basic info
print("First few rows:")
print(df.head())
print("\nData columns:", df.columns.tolist())

# ------------------------------
# 2. Prepare datetime and time-of-day
# ------------------------------
# Check if we have both 'date' and 'time' columns
if 'date' not in df.columns:
    raise ValueError("CSV does not contain a 'date' column.")
if 'time' not in df.columns:
    print("Warning: 'time' column not found. Attempting to parse from 'date' (if it contains time).")
    # If 'date' contains both date and time, try to parse it directly.
    # But based on your extractor, time is separate. We'll assume it's missing and raise.
    raise ValueError("Time column is required for this analysis. Please ensure your CSV includes a 'time' column.")

# Convert date to datetime
df['parsed_date'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')

# Drop rows with invalid dates
initial_len = len(df)
df = df.dropna(subset=['parsed_date'])
if len(df) < initial_len:
    print(f"Dropped {initial_len - len(df)} rows with invalid dates.")

# Combine date and time to a single timestamp
# The time format in your extractor is like "3:45 PM" or "3:45PM". We'll handle common variations.
def combine_date_time(row):
    try:
        time_str = str(row['time']).strip()
        # Try parsing with various formats
        for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M"]:
            try:
                time_obj = pd.to_datetime(time_str, format=fmt).time()
                return pd.Timestamp.combine(row['parsed_date'].date(), time_obj)
            except:
                continue
        # If all fail, return NaT
        return pd.NaT
    except:
        return pd.NaT

df['timestamp'] = df.apply(combine_date_time, axis=1)
df = df.dropna(subset=['timestamp'])  # drop rows where time parsing failed

# Extract hour of day
df['hour'] = df['timestamp'].dt.hour

# ------------------------------
# 3. Filter for spending (DEBIT)
# ------------------------------
if 'type' in df.columns:
    spending_df = df[df['type'].str.upper().str.contains('DEBIT')].copy()
else:
    # If no type column, assume all are spending? Better to warn.
    print("Warning: No 'type' column found. Assuming all transactions are spending.")
    spending_df = df.copy()

if spending_df.empty:
    print("No spending transactions found. Exiting.")
    exit()

# ------------------------------
# 4. Group by hour and analyze
# ------------------------------
hourly_stats = spending_df.groupby('hour')['amount'].agg(['sum', 'count', 'mean']).reset_index()
hourly_stats.columns = ['hour', 'total_spent', 'transaction_count', 'avg_transaction']

# Ensure all hours 0-23 are present (fill missing with zeros)
hourly_stats = hourly_stats.set_index('hour').reindex(range(24), fill_value=0).reset_index()

# ------------------------------
# 5. Visualize
# ------------------------------
# Bar chart of total spending per hour
plt.figure()
plt.bar(hourly_stats['hour'], hourly_stats['total_spent'], color='skyblue', edgecolor='black')
plt.xlabel('Hour of Day')
plt.ylabel('Total Amount Spent (₹)')
plt.title('Total Spending by Hour of Day (One Month)')
plt.xticks(range(0, 24, 2))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line plot of average transaction value per hour
plt.figure()
plt.plot(hourly_stats['hour'], hourly_stats['avg_transaction'], marker='o', linestyle='-', color='orange')
plt.xlabel('Hour of Day')
plt.ylabel('Average Transaction Amount (₹)')
plt.title('Average Transaction Value by Hour of Day')
plt.xticks(range(0, 24, 2))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Optional: Heatmap of spending by hour and day of week? But user asked one-month period, so we can keep simple.

# ------------------------------
# 6. Print summary table
# ------------------------------
print("\nSpending by Hour of Day:")
print(hourly_stats.to_string(index=False))

# Save to CSV for further use
hourly_stats.to_csv('hourly_spending_summary.csv', index=False)
print("\nHourly summary saved to 'hourly_spending_summary.csv'")