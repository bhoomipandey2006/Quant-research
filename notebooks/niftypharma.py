import pandas as pd

#Load the data set
df = pd.read_csv("../datasets/NIFTY PHARMA.csv")
print(df.head()) #print first 5 readings
print(df.info()) 
print(df.describe())

df['Date'] = pd.to_datetime(df['Date']) #Convert date into datetime
df['return'] = df['Close'].pct_change()*100 #Formula to calculate return


df['day_of_week'] = df['Date'].dt.day_name() #Define day of week

print(df[['Date', 'day_of_week','return']].tail(35)) #Print date, day of week & return of last 35 readings

avg_by_day = df.groupby('day_of_week')['return'].mean() #Find the average return by grouping day of week
print(avg_by_day)

df = df[~df['day_of_week'].isin(['Saturday', 'Sunday'])] #Remove weekend data quality issue

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