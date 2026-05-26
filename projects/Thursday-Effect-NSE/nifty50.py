import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

#Load the data set
df = pd.read_csv("../../datasets/nifty50.csv")
print(df.head()) #print first 5 readings
print(df.info()) 
print(df.describe())

df['Date'] = pd.to_datetime(df['Date']) #Convert date into datetime
df['return'] = df['Close'].pct_change()*100 #Formula to calculate return


df['day_of_week'] = df['Date'].dt.day_name() #Define day of week

print(df[['Date', 'day_of_week','return']].tail(35)) #Print date, day of week & return of last 35 readings

avg_by_day = df.groupby('day_of_week')['return'].mean() #Find the average return by grouping day of week
print(avg_by_day)

df = df[~df['day_of_week'].isin(['Saturday', 'Sunday'])].copy() #Remove weekend data quality issue

avg_by_day = df.groupby('day_of_week')['return'].mean() #Recalculate

order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] #Order by actual weekday
print(avg_by_day.reindex(order))

positive_rate = df.groupby('day_of_week')['return'].apply( 
    lambda x: (x > 0).sum() / len(x) * 100
)   #Count how many of each day is positive
print(positive_rate.reindex(order))

# 1. Mark Thursdays (3 equals Thursday)
df['is_thursday'] = df['Date'].dt.weekday == 3

# 2. Mark 1 day before Thursday (Wednesday) and 2 days before (Tuesday)
df['one_day_before'] = df['is_thursday'].shift(-1, fill_value=False)
df['two_days_before'] = df['is_thursday'].shift(-2, fill_value=False)

# 3. Combine them into a single "Pre-Expiry" flag (Tuesday OR Wednesday)
df['pre_expiry'] = df['one_day_before'] | df['two_days_before']
# Group by the flag and calculate the average return
analysis = df.groupby('pre_expiry')['return'].mean()

print(analysis)


def analyze_sector(df, sector_name):
    df['Date'] = pd.to_datetime(df['Date'])
    df['return'] = df['Close'].pct_change() * 100
    df['day_of_week'] = df['Date'].dt.day_name()
    df = df[~df['day_of_week'].isin(['Saturday', 'Sunday'])].copy()
    df['is_thursday'] = df['Date'].dt.weekday == 3
    df['one_day_before'] = df['is_thursday'].shift(-1, fill_value=False)
    df['two_days_before'] = df['is_thursday'].shift(-2, fill_value=False)
    df['pre_expiry'] = df['one_day_before'] | df['two_days_before']
    
    pre = df[df['pre_expiry'] == True]['return'].dropna()
    normal = df[df['pre_expiry'] == False]['return'].dropna()
    
    t_stat, p_value = stats.ttest_ind(pre, normal)
    
    print(f"\n{sector_name}:")
    print(f"Pre-expiry avg: {pre.mean():.4f}%")
    print(f"Normal avg:     {normal.mean():.4f}%")
    print(f"P-value:        {p_value:.4f}")
    print(f"Significant:    {p_value < 0.05}")

analyze_sector(pd.read_csv("../../datasets/nifty50.csv"), "Nifty 50")
analyze_sector(pd.read_csv("../../datasets/NIFTY IT.csv"), "Nifty IT")
analyze_sector(pd.read_csv("../../datasets/NIFTY PHARMA.csv"), "Nifty Pharma")
analyze_sector(pd.read_csv("../../datasets/NIFTY BANK.csv"), "Nifty Bank")
analyze_sector(pd.read_csv("../../datasets/NIFTY METAL.csv"), "Nifty Metal")
analyze_sector(pd.read_csv("../../datasets/NIFTY FMCG.csv"), "Nifty FMCG")


sectors = ['Nifty 50', 'Nifty IT', 'Nifty Pharma',
           'Nifty Bank', 'Nifty Metal', 'Nifty FMCG']
normal =     [0.029, -0.024, 0.041, 0.064, 0.002, 0.043]
pre_expiry = [0.088,  0.136, 0.085, 0.114, 0.072, 0.074]
p_values =   [0.11,   0.02,  0.24,  0.35,  0.34,  0.41]

x = np.arange(len(sectors))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, normal, width,
               label='Normal Days', color='#94a3b8', alpha=0.85)
bars2 = ax.bar(x + width/2, pre_expiry, width,
               label='Pre-Expiry Days', color='#6C63FF', alpha=0.85)

# Mark significant result
ax.annotate('★ p=0.02\nStatistically\nSignificant',
            xy=(1 + width/2, 0.136),
            xytext=(1.6, 0.16),
            fontsize=9, color='#6C63FF', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#6C63FF'))

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(sectors, rotation=10)
ax.set_ylabel('Average Daily Return (%)')
ax.set_title(
    'Pre-Expiry Effect Across 6 NSE Sectors (2000–2024)\n'
    'Only Nifty IT shows statistical significance (p=0.02)',
    fontsize=12, fontweight='bold'
)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('expiry_effect_chart.png', dpi=150)
print("Chart saved.")