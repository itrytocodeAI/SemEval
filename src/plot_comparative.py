import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Load data
df = pd.read_csv('logs/v3Final_comparative.csv')
os.makedirs('plots/comparative', exist_ok=True)

# -------------------------------------------------------------
# 1. Bubble Plot (X=PCC_V, Y=PCC_A, Size=1/RMSE)
# -------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Smaller RMSE = Larger Bubble (better)
df['Bubble_Size'] = (1.0 / df['RMSE_VA']) * 1000 

palette = {"Baseline": "gray", "Transformer": "orange", "Proposed": "green"}

for typ in df['Type'].unique():
    subset = df[df['Type'] == typ]
    plt.scatter(
        subset['PCC_V'], subset['PCC_A'], 
        s=subset['Bubble_Size'], 
        c=palette[typ], label=typ, 
        alpha=0.6, edgecolors="white", linewidth=2
    )

for i in range(len(df)):
    if 'V3_Phase3' in df['Model_Version'].iloc[i] or 'V2_' in df['Model_Version'].iloc[i]:
        plt.text(df['PCC_V'].iloc[i], df['PCC_A'].iloc[i] + 0.02, 
                 df['Model_Version'].iloc[i] + " (" + df['Domain'].iloc[i][:4] + ")", 
                 fontsize=9, ha='center', weight='bold')

plt.title('Comparative Analysis: Valence vs Arousal Correlation (Bubble = 1/RMSE)', fontsize=14)
plt.xlabel('Pearson r (Valence) [Higher is Better]', fontsize=12)
plt.ylabel('Pearson r (Arousal) [Higher is Better]', fontsize=12)
plt.legend(scatterpoints=1, borderpad=1, labelspacing=1, title='Architecture Type')
plt.tight_layout()
plt.savefig('plots/comparative/bubble_plot_v3_sota.png', dpi=300)
plt.close()


# -------------------------------------------------------------
# 2. Radar Plot (Normalized Metrics for V3_Phase3 vs V2_DeBERTa vs BERT)
# -------------------------------------------------------------
def plot_radar(domain):
    domain_df = df[df['Domain'] == domain].copy()
    
    # Filter only key models for clean radar
    models_to_plot = ['BERT', 'V2_DeBERTa', 'V3_Phase3_SOTA']
    domain_df = domain_df[domain_df['Model_Version'].isin(models_to_plot)]
    
    # Normalize RMSE: 2.0 - RMSE (since max RMSE in base ~ 1.5, smaller RMSE -> higher score ~1.15)
    domain_df['Inv_RMSE'] = 2.0 - domain_df['RMSE_VA']
    
    categories = ['PCC_V', 'PCC_A', 'Inv_RMSE']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], ['Valence\nCorrelation', 'Arousal\nCorrelation', 'Inverse RMSE\n(Higher is Better)'])
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.5, 0.8, 1.1], ["0.2", "0.5", "0.8", "1.1"], color="grey", size=8)
    plt.ylim(0, 1.25)
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model in enumerate(models_to_plot):
        row = domain_df[domain_df['Model_Version'] == model]
        if row.empty: continue
            
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        
    plt.title(f'Radar Analysis ({domain.capitalize()})', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(f'plots/comparative/radar_plot_{domain}.png', dpi=300)
    plt.close()

plot_radar('Restaurant')
plot_radar('Laptop')

print("Comparative analysis plots successfully generated in plots/comparative/")
