import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load race results ──────────────────────────────────────
race_files = {
    2019: 'formula1_2019season_raceResults.csv',
    2020: 'formula1_2020season_raceResults.csv',
    2021: 'formula1_2021season_raceResults.csv',
    2022: 'Formula1_2022season_raceResults.csv',
    2023: 'Formula1_2023season_raceResults.csv',
    2024: 'Formula1_2024season_raceResults.csv',
    2025: 'Formula1_2025Season_RaceResults.csv'
}

quali_files = {
    2022: 'Formula1_2022season_qualifyingResults.csv',
    2023: 'Formula1_2023season_qualifyingResults.csv',
    2024: 'Formula1_2024season_qualifyingResults.csv',
    2025: 'Formula1_2025Season_QualifyingResults.csv'
}

# Load races
race_dfs = []
for year, filename in race_files.items():
    try:
        temp = pd.read_csv(filename)
        temp['Year'] = year
        race_dfs.append(temp)
        print(f"Loaded races {year}: {len(temp)} rows")
    except FileNotFoundError:
        print(f"Missing: {filename}")

df = pd.concat(race_dfs, ignore_index=True)

# Load qualifying
quali_dfs = []
for year, filename in quali_files.items():
    try:
        temp = pd.read_csv(filename)
        temp['Year'] = year
        quali_dfs.append(temp)
    except FileNotFoundError:
        print(f"Missing quali: {filename}")

quali_df = pd.concat(quali_dfs, ignore_index=True)

# ── STEP 2: Clean race data ────────────────────────────────────────
df['Position_Clean'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position_Clean']).copy()
df['Position_Clean'] = df['Position_Clean'].astype(int)
df['Podium'] = (df['Position_Clean'] <= 3).astype(int)
df['Starting_Grid'] = pd.to_numeric(df['Starting Grid'], errors='coerce')
df['Starting_Grid'] = df['Starting_Grid'].fillna(df['Starting_Grid'].median())

# ── STEP 3: Merge qualifying position ─────────────────────────────
quali_clean = quali_df[['Track', 'Driver', 'Year', 'Position']].copy()
quali_clean.columns = ['Track', 'Driver', 'Year', 'Quali_Position']
quali_clean['Quali_Position'] = pd.to_numeric(
    quali_clean['Quali_Position'], errors='coerce'
)
df = df.merge(quali_clean, on=['Track', 'Driver', 'Year'], how='left')
df['Quali_Position'] = df['Quali_Position'].fillna(df['Starting_Grid'])

print(f"\nTotal rows: {len(df)}")
print(f"Overall podium rate: {df['Podium'].mean()*100:.1f}%")

# ── STEP 4: Sort by driver and year for time-based features ────────
df = df.sort_values(['Driver', 'Year', 'Track']).reset_index(drop=True)

# ── STEP 5: Rolling form — last 3 races (recent form matters more) ─
df['Rolling_Form_3'] = df.groupby('Driver')['Podium'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
).fillna(0)

# ── STEP 6: Season podium rate for this driver in this year ────────
season_rates = df.groupby(['Driver', 'Year'])['Podium'].transform('mean')
df['Season_Podium_Rate'] = season_rates

# ── STEP 7: Circuit-specific historical podium rate per driver ─────
# e.g. Hamilton wins Monaco a lot — this captures that
circuit_rates = df.groupby(['Driver', 'Track'])['Podium'].transform('mean')
df['Circuit_Podium_Rate'] = circuit_rates

# ── STEP 8: Encode categoricals ───────────────────────────────────
le_driver = LabelEncoder()
le_track  = LabelEncoder()
le_team   = LabelEncoder()

df['Driver_Enc'] = le_driver.fit_transform(df['Driver'])
df['Track_Enc']  = le_track.fit_transform(df['Track'])
df['Team_Enc']   = le_team.fit_transform(df['Team'])

# ── STEP 9: Train model ────────────────────────────────────────────
features = [
    'Driver_Enc',
    'Track_Enc',
    'Team_Enc',
    'Starting_Grid',
    'Quali_Position',       # how well they qualified
    'Rolling_Form_3',       # last 3 races form
    'Season_Podium_Rate',   # how good is their season so far
    'Circuit_Podium_Rate',  # their history at this specific track
    'Year'
]

X = df[features]
y = df['Podium']

model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Model accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── STEP 10: Show what's driving predictions ───────────────────────
print("\n--- Recent form (2023-2025) ---")
recent = df[df['Year'] >= 2023]
for drv in ['Gabriel Bortoleto', 'Max Verstappen', 'Kimi Antonelli']:
    d = recent[recent['Driver'] == drv]
    if len(d) > 0:
        rate = d['Podium'].mean() * 100
        pods = d['Podium'].sum()
        races = len(d)
        print(f"{drv}: {pods}/{races} podiums = {rate:.1f}%")

print("\n--- Circuit history (all years) ---")
for drv in ['Gabriel Bortoleto', 'Max Verstappen', 'Kimi Antonelli']:
    for trk in ['Monaco', 'Italy', 'Great Britain']:
        d = df[(df['Driver'] == drv) & (df['Track'] == trk)]
        if len(d) > 0:
            rate = d['Podium'].mean() * 100
            print(f"{drv} at {trk}: {rate:.0f}% ({d['Podium'].sum()}/{len(d)})")

# ── STEP 11: Prediction function ──────────────────────────────────
def predict_podium(driver_name, track_name, quali_pos, year=2025):
    if driver_name not in le_driver.classes_:
        print(f"Driver not found: '{driver_name}'")
        return None
    if track_name not in le_track.classes_:
        print(f"Track not found: '{track_name}'")
        print(f"Available: {sorted(le_track.classes_)}")
        return None

    driver_rows = df[df['Driver'] == driver_name]
    if len(driver_rows) == 0:
        return None

    # Most recent team
    recent_team = driver_rows.sort_values('Year').iloc[-1]['Team']
    if recent_team not in le_team.classes_:
        recent_team = df['Team'].mode()[0]

    # Recent 3-race form
    recent_form = driver_rows['Rolling_Form_3'].iloc[-1]

    # Season podium rate — use 2025 if available, else most recent
    season_rows = driver_rows[driver_rows['Year'] == year]
    season_rate = season_rows['Season_Podium_Rate'].mean() if len(season_rows) > 0 \
        else driver_rows[driver_rows['Year'] == driver_rows['Year'].max()]['Season_Podium_Rate'].mean()

    # Circuit-specific rate for this driver
    circuit_rows = driver_rows[driver_rows['Track'] == track_name]
    circuit_rate = circuit_rows['Circuit_Podium_Rate'].mean() if len(circuit_rows) > 0 \
        else season_rate

    input_row = pd.DataFrame([{
        'Driver_Enc':          le_driver.transform([driver_name])[0],
        'Track_Enc':           le_track.transform([track_name])[0],
        'Team_Enc':            le_team.transform([recent_team])[0],
        'Starting_Grid':       quali_pos,
        'Quali_Position':      quali_pos,
        'Rolling_Form_3':      recent_form,
        'Season_Podium_Rate':  season_rate,
        'Circuit_Podium_Rate': circuit_rate,
        'Year':                year
    }])

    prob = model.predict_proba(input_row)[0][1]
    return prob

# ── STEP 12: Predictions ───────────────────────────────────────────
drivers = ['Gabriel Bortoleto', 'Max Verstappen', 'Kimi Antonelli']
tracks  = ['Monaco', 'Spain', 'Canada', 'Great Britain', 'Italy']

print("\n" + "="*62)
print("F1 PODIUM PROBABILITY — LOGISTIC REGRESSION · 2025")
print("="*62)

results = []
for driver in drivers:
    print(f"\n{driver}:")
    for track in tracks:
        prob = predict_podium(driver, track, quali_pos=3)
        if prob is not None:
            print(f"  {track:<20} P3 quali → {prob*100:.1f}%")
            results.append({
                'Driver': driver,
                'Track': track,
                'Podium_Probability_%': round(prob * 100, 1)
            })

# ── STEP 13: Custom scenarios ──────────────────────────────────────
print("\n--- Custom qualifying scenarios ---")
custom = [
    ('Sebastian Vettel',  'Monaco',        1),
    ('Max Verstappen',  'Monaco',        1),
    ('Fernando Alonso', 'Monaco',        3),
    ('Sebastian Vettel',  'Italy',         5),
    ('Max Verstappen',  'Italy',         1),
    ('Sebastian Vettel',  'Great Britain', 1),
    ('Fernando Alonso', 'Spain',         2),
]
for driver, track, grid in custom:
    prob = predict_podium(driver, track, grid)
    if prob:
        print(f"{driver:<22} | {track:<16} | P{grid} → {prob*100:.1f}%")

# ── STEP 14: Chart ─────────────────────────────────────────────────
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(13, 7))
colors = {
    'Gabriel Bortoleto':  '#00D2BE',
    'Max Verstappen':  '#1E41FF',
    'Kimi Antonelli': '#006F62'
}

x     = np.arange(len(tracks))
width = 0.25

for i, driver in enumerate(drivers):
    d = results_df[results_df['Driver'] == driver]
    probs = []
    for t in tracks:
        val = d[d['Track'] == t]['Podium_Probability_%'].values
        probs.append(val[0] if len(val) > 0 else 0)
    bars = ax.bar(x + i * width, probs, width,
                  label=driver, color=colors[driver], alpha=0.85)
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{prob:.0f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.4,
           linewidth=1, label='33% baseline')
ax.set_xlabel('Circuit', fontsize=11)
ax.set_ylabel('Podium Probability (%)', fontsize=11)
ax.set_title(
    'F1 Podium Probability — Hamilton vs Verstappen vs Alonso\n'
    'Logistic Regression · Circuit history + recent form + qualifying position',
    fontsize=12, fontweight='bold', pad=15
)
ax.set_xticks(x + width)
ax.set_xticklabels(tracks, rotation=10, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('f1_podium_probability.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved: f1_podium_probability.png")

# ── STEP 15: Export Excel ──────────────────────────────────────────
results_df.to_excel('f1_predictions.xlsx', index=False)
print("Excel saved: f1_predictions.xlsx")
print("\nDone.")