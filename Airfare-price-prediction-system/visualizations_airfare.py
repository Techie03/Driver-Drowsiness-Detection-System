"""
Airfare Price Prediction - Visualizations
==========================================
Generate comprehensive visualizations for flight price prediction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 12)

print("Generating visualizations for Airfare Price Prediction Analysis...")

# ============================================================================
# Load saved results
# ============================================================================

try:
    model_comparison = pd.read_csv('/home/claude/model_comparison.csv', index_col=0)
    predictions = pd.read_csv('/home/claude/predictions.csv')
    
    try:
        feature_importance = pd.read_csv('/home/claude/feature_importance.csv')
        has_feature_importance = True
    except:
        has_feature_importance = False
        print("⚠ Feature importance file not found")
    
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"⚠ Error loading data: {e}")
    print("⚠ Run airfare_price_prediction.py first to generate data")
    exit()

# ============================================================================
# Create main visualization dashboard
# ============================================================================

fig = plt.figure(figsize=(20, 14))

# 1. Model Accuracy Comparison
ax1 = plt.subplot(3, 3, 1)
models = model_comparison.index
accuracies = model_comparison['Accuracy']
colors = ['#FF6B6B' if acc < 85 else '#4ECDC4' for acc in accuracies]
bars = ax1.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 100])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)
ax1.axvline(x=88, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 88%')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# 2. R² Score Comparison
ax2 = plt.subplot(3, 3, 2)
r2_scores = model_comparison['R2']
bars = ax2.bar(range(len(models)), r2_scores, color='#4ECDC4', 
               edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('R² Score by Model', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylim([0, 1])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# 3. RMSE Comparison
ax3 = plt.subplot(3, 3, 3)
rmse_values = model_comparison['RMSE']
bars = ax3.bar(range(len(models)), rmse_values, color='#FF6B6B', 
               edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_ylabel('RMSE (₹)', fontsize=12, fontweight='bold')
ax3.set_title('Root Mean Squared Error by Model', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
            f'₹{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# 4. Actual vs Predicted Scatter Plot
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(predictions['Actual_Price'], predictions['Predicted_Price'], 
           alpha=0.6, s=50, c='#4ECDC4', edgecolors='black', linewidth=0.5)
min_price = min(predictions['Actual_Price'].min(), predictions['Predicted_Price'].min())
max_price = max(predictions['Actual_Price'].max(), predictions['Predicted_Price'].max())
ax4.plot([min_price, max_price], [min_price, max_price], 
        'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
ax4.set_xlabel('Actual Price (₹)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
ax4.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Calculate R² for this plot
r2 = r2_score(predictions['Actual_Price'], predictions['Predicted_Price'])
ax4.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax4.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 5. Residual Plot
ax5 = plt.subplot(3, 3, 5)
residuals = predictions['Predicted_Price'] - predictions['Actual_Price']
ax5.scatter(predictions['Predicted_Price'], residuals, 
           alpha=0.6, s=50, c='#FF6B6B', edgecolors='black', linewidth=0.5)
ax5.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_xlabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Residuals (₹)', fontsize=12, fontweight='bold')
ax5.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Error Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(predictions['Absolute_Error'], bins=30, color='#4ECDC4', 
        edgecolor='black', alpha=0.7)
ax6.set_xlabel('Absolute Error (₹)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
mean_error = predictions['Absolute_Error'].mean()
ax6.axvline(x=mean_error, color='red', linestyle='--', linewidth=2, 
           label=f'Mean Error: ₹{mean_error:.0f}')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Feature Importance (if available)
if has_feature_importance:
    ax7 = plt.subplot(3, 3, 7)
    top_features = feature_importance.head(10)
    colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax7.barh(top_features['feature'], top_features['importance'], 
                   color=colors_feat, edgecolor='black', linewidth=1)
    ax7.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax7.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    ax7.invert_yaxis()
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax7.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
else:
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.5, 0.5, 'Feature Importance\nNot Available', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax7.axis('off')

# 8. MAE Comparison
ax8 = plt.subplot(3, 3, 8)
mae_values = model_comparison['MAE']
bars = ax8.bar(range(len(models)), mae_values, color='#95E1D3', 
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax8.set_ylabel('MAE (₹)', fontsize=12, fontweight='bold')
ax8.set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold')
ax8.set_xticks(range(len(models)))
ax8.set_xticklabels(models, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
            f'₹{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax8.grid(axis='y', alpha=0.3)

# 9. Percentage Error Distribution
ax9 = plt.subplot(3, 3, 9)
ax9.hist(predictions['Percentage_Error'], bins=30, color='#F38181', 
        edgecolor='black', alpha=0.7)
ax9.set_xlabel('Percentage Error (%)', fontsize=12, fontweight='bold')
ax9.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax9.set_title('Distribution of Percentage Errors', fontsize=14, fontweight='bold')
mean_pct_error = predictions['Percentage_Error'].mean()
ax9.axvline(x=mean_pct_error, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_pct_error:.1f}%')
ax9.axvline(x=10, color='green', linestyle='--', linewidth=2, alpha=0.5,
           label='10% threshold')
ax9.legend()
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/airfare_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Main visualizations saved to: airfare_visualizations.png")

# ============================================================================
# Create additional detailed plots
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Metrics Comparison Heatmap
ax1 = axes[0, 0]
metrics_for_heatmap = model_comparison[['R2', 'MAE', 'RMSE', 'MAPE']].copy()
# Normalize metrics for better visualization
metrics_normalized = (metrics_for_heatmap - metrics_for_heatmap.min()) / (metrics_for_heatmap.max() - metrics_for_heatmap.min())
sns.heatmap(metrics_normalized.T, annot=True, fmt='.3f', cmap='RdYlGn', 
           cbar_kws={'label': 'Normalized Score'}, ax=ax1,
           xticklabels=model_comparison.index, linewidths=1, linecolor='black')
ax1.set_title('Model Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_xlabel('Models', fontsize=12, fontweight='bold')

# 2. Error Box Plot
ax2 = axes[0, 1]
error_data = [predictions['Absolute_Error']]
bp = ax2.boxplot(error_data, labels=['Absolute Error'], patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(facecolor='#4ECDC4', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='blue', linewidth=2, linestyle='--'))
ax2.set_ylabel('Error (₹)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add statistics text
q1 = predictions['Absolute_Error'].quantile(0.25)
median = predictions['Absolute_Error'].median()
q3 = predictions['Absolute_Error'].quantile(0.75)
mean = predictions['Absolute_Error'].mean()
ax2.text(1.2, q1, f'Q1: ₹{q1:.0f}', fontsize=10, fontweight='bold')
ax2.text(1.2, median, f'Median: ₹{median:.0f}', fontsize=10, fontweight='bold')
ax2.text(1.2, q3, f'Q3: ₹{q3:.0f}', fontsize=10, fontweight='bold')
ax2.text(1.2, mean, f'Mean: ₹{mean:.0f}', fontsize=10, fontweight='bold', color='blue')

# 3. Model Comparison Bar Chart (All Metrics)
ax3 = axes[1, 0]
x = np.arange(len(model_comparison))
width = 0.2

# Normalize for visualization
accuracy_norm = model_comparison['Accuracy'] / 100
r2_norm = model_comparison['R2']

bars1 = ax3.bar(x - width, accuracy_norm, width, label='Accuracy', 
               color='#4ECDC4', edgecolor='black', alpha=0.8)
bars2 = ax3.bar(x, r2_norm, width, label='R²', 
               color='#95E1D3', edgecolor='black', alpha=0.8)

ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy and R² Score Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(model_comparison.index, rotation=45, ha='right')
ax3.legend()
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)

# 4. Prediction Accuracy by Price Range
ax4 = axes[1, 1]
# Bin predictions by price range
price_bins = pd.cut(predictions['Actual_Price'], bins=5)
accuracy_by_bin = predictions.groupby(price_bins)['Percentage_Error'].mean()

bin_labels = [f'₹{int(interval.left)}-{int(interval.right)}' for interval in accuracy_by_bin.index]
colors_bins = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bin_labels)))

bars = ax4.bar(range(len(bin_labels)), accuracy_by_bin.values, 
              color=colors_bins, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('Price Range', fontsize=12, fontweight='bold')
ax4.set_ylabel('Mean Percentage Error (%)', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Accuracy by Price Range', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(bin_labels)))
ax4.set_xticklabels(bin_labels, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Detailed analysis plots saved to: detailed_analysis.png")

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. airfare_visualizations.png - Main dashboard (9 subplots)")
print("  2. detailed_analysis.png - Additional insights (4 subplots)")
