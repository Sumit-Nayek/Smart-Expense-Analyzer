import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for better styling

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Load the CSV file (adjust filename as needed)
csv_file = "/content/PhonePe_Statement_Jan2026_Feb2026_transactions.csv"  # replace with your file
df = pd.read_csv(csv_file)

# 2. Inspect the data
print("First few rows:")
print(df.head())
print("\nData info:")
print(df.info())

# 3. Convert the 'date' column to datetime
# The date format in the CSV is like "Feb 13, 2026"
df['parsed_date'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')

# Check for any parsing errors
if df['parsed_date'].isna().any():
    print(f"Warning: {df['parsed_date'].isna().sum()} dates could not be parsed.")
    # Optionally drop rows with invalid dates
    df = df.dropna(subset=['parsed_date'])

# 4. Add a 'week' column (starting Monday)
df['week_start'] = df['parsed_date'].dt.to_period('W').dt.start_time

# 5. Filter for DEBIT transactions (spending)
spending_df = df[df['type'] == 'DEBIT'].copy()

if spending_df.empty:
    print("No DEBIT transactions found. Check the 'type' column values.")
else:
    # 6. Group by week and sum the amounts
    weekly_spending = spending_df.groupby('week_start')['amount'].agg(['sum', 'count', 'mean']).reset_index()
    weekly_spending.columns = ['week_start', 'total_spent', 'transaction_count', 'average_transaction']

    # Sort by week
    weekly_spending = weekly_spending.sort_values('week_start')

    print("\nWeekly Spending Summary:")
    print(weekly_spending.to_string(index=False))

    # 7. Plot weekly spending
    plt.figure()
    plt.plot(weekly_spending['week_start'], weekly_spending['total_spent'], marker='o', linestyle='-', color='b')
    plt.title('Weekly Spending (Total Amount)')
    plt.xlabel('Week Starting')
    plt.ylabel('Total Amount (₹)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Optional: Bar chart of weekly spending
    plt.figure()
    plt.bar(weekly_spending['week_start'].astype(str), weekly_spending['total_spent'], color='coral')
    plt.title('Weekly Spending (Bar Chart)')
    plt.xlabel('Week Starting')
    plt.ylabel('Total Amount (₹)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 8. Optional: Top recipients per week
    top_per_week = spending_df.groupby(['week_start', 'recipient'])['amount'].sum().reset_index()
    top_per_week = top_per_week.sort_values(['week_start', 'amount'], ascending=[True, False])
    print("\nTop recipient by spending each week:")
    print(top_per_week.groupby('week_start').first().reset_index())

# 9. (Optional) Save the weekly summary to CSV
weekly_spending.to_csv('weekly_spending_summary.csv', index=False)
print("\nWeekly summary saved to 'weekly_spending_summary.csv'")