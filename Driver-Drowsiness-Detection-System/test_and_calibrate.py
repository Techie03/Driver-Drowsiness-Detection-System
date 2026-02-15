"""
Driver Drowsiness Detection - Testing & Calibration
====================================================
Analyze detection performance and calibrate thresholds using test data
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

class Config:
    """Configuration parameters"""
    EAR_THRESHOLD = 0.16
    EAR_CONSEC_FRAMES = 90
    TARGET_FPS = 30

class DetectionTester:
    """Test and calibrate drowsiness detection system"""
    
    def __init__(self):
        self.test_results = []
        self.ear_values = []
        self.states = []
        self.timestamps = []
        
    def simulate_ear_sequence(self, duration_seconds=60):
        """
        Simulate realistic EAR values for testing
        Simulates: alert periods, blink events, drowsy periods
        """
        fps = 30
        total_frames = duration_seconds * fps
        
        ear_sequence = []
        state_sequence = []
        
        # Normal alert EAR range: 0.25 - 0.30
        # Blink: drops to ~0.10 for 3-5 frames
        # Drowsy: sustained below 0.16 for 90+ frames
        
        frame = 0
        while frame < total_frames:
            # Decide on sequence type
            rand = np.random.random()
            
            if rand < 0.7:  # 70% alert state
                # Alert period (5-10 seconds)
                period_frames = np.random.randint(5 * fps, 10 * fps)
                for _ in range(period_frames):
                    if frame >= total_frames:
                        break
                    
                    # Normal alert EAR with small variations
                    ear = np.random.uniform(0.25, 0.32)
                    
                    # Random blinks (every 3-5 seconds on average)
                    if np.random.random() < 0.007:  # ~0.7% chance per frame
                        # Blink sequence (3-5 frames)
                        blink_frames = np.random.randint(3, 6)
                        for _ in range(blink_frames):
                            if frame >= total_frames:
                                break
                            ear_sequence.append(np.random.uniform(0.08, 0.12))
                            state_sequence.append('alert')
                            frame += 1
                        continue
                    
                    ear_sequence.append(ear)
                    state_sequence.append('alert')
                    frame += 1
            
            elif rand < 0.9:  # 20% transition/micro-sleep
                # Brief drowsy period (2-4 seconds)
                period_frames = np.random.randint(2 * fps, 4 * fps)
                for _ in range(period_frames):
                    if frame >= total_frames:
                        break
                    ear = np.random.uniform(0.13, 0.18)
                    ear_sequence.append(ear)
                    state_sequence.append('drowsy' if ear < 0.16 else 'alert')
                    frame += 1
            
            else:  # 10% true drowsy state
                # Prolonged drowsy period (5-10 seconds)
                period_frames = np.random.randint(5 * fps, 10 * fps)
                for _ in range(period_frames):
                    if frame >= total_frames:
                        break
                    ear = np.random.uniform(0.10, 0.15)
                    ear_sequence.append(ear)
                    state_sequence.append('drowsy')
                    frame += 1
        
        return np.array(ear_sequence[:total_frames]), state_sequence[:total_frames]
    
    def test_threshold_sensitivity(self, ear_data, ground_truth, threshold_range=(0.12, 0.25)):
        """
        Test different EAR thresholds and calculate metrics
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
        results = []
        
        for threshold in thresholds:
            # Simulate detection with this threshold
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            consecutive_drowsy = 0
            detected_drowsy = False
            
            for i, (ear, true_state) in enumerate(zip(ear_data, ground_truth)):
                # Simulate 3-second (90 frame) threshold
                if ear < threshold:
                    consecutive_drowsy += 1
                    if consecutive_drowsy >= 90:
                        detected_drowsy = True
                else:
                    consecutive_drowsy = 0
                    detected_drowsy = False
                
                # Calculate confusion matrix
                if detected_drowsy and true_state == 'drowsy':
                    true_positives += 1
                elif detected_drowsy and true_state == 'alert':
                    false_positives += 1
                elif not detected_drowsy and true_state == 'alert':
                    true_negatives += 1
                elif not detected_drowsy and true_state == 'drowsy':
                    false_negatives += 1
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(ear_data)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
        
        return pd.DataFrame(results)
    
    def analyze_frame_count_impact(self, ear_data, ground_truth, threshold=0.16):
        """
        Analyze impact of consecutive frame count on detection
        """
        frame_counts = range(30, 180, 15)  # 1 to 6 seconds at 30 FPS
        results = []
        
        for frame_count in frame_counts:
            true_positives = 0
            false_positives = 0
            
            consecutive_drowsy = 0
            detected_drowsy = False
            
            for ear, true_state in zip(ear_data, ground_truth):
                if ear < threshold:
                    consecutive_drowsy += 1
                    if consecutive_drowsy >= frame_count:
                        detected_drowsy = True
                else:
                    consecutive_drowsy = 0
                    detected_drowsy = False
                
                if detected_drowsy and true_state == 'drowsy':
                    true_positives += 1
                elif detected_drowsy and true_state == 'alert':
                    false_positives += 1
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            results.append({
                'frame_count': frame_count,
                'seconds': frame_count / 30,
                'precision': precision,
                'false_alarms': false_positives
            })
        
        return pd.DataFrame(results)
    
    def visualize_results(self, ear_data, ground_truth, threshold_df, frame_df):
        """
        Create comprehensive visualization of test results
        """
        fig = plt.figure(figsize=(20, 12))
        
        # 1. EAR Time Series
        ax1 = plt.subplot(3, 3, 1)
        time_seconds = np.arange(len(ear_data)) / 30
        colors = ['red' if s == 'drowsy' else 'green' for s in ground_truth]
        ax1.scatter(time_seconds, ear_data, c=colors, alpha=0.5, s=1)
        ax1.axhline(y=0.16, color='blue', linestyle='--', linewidth=2, label='Threshold (0.16)')
        ax1.set_xlabel('Time (seconds)', fontweight='bold')
        ax1.set_ylabel('EAR Value', fontweight='bold')
        ax1.set_title('EAR Values Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. EAR Distribution
        ax2 = plt.subplot(3, 3, 2)
        alert_ears = [ear for ear, state in zip(ear_data, ground_truth) if state == 'alert']
        drowsy_ears = [ear for ear, state in zip(ear_data, ground_truth) if state == 'drowsy']
        
        ax2.hist(alert_ears, bins=50, alpha=0.6, label='Alert', color='green', edgecolor='black')
        ax2.hist(drowsy_ears, bins=50, alpha=0.6, label='Drowsy', color='red', edgecolor='black')
        ax2.axvline(x=0.16, color='blue', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('EAR Value', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('EAR Distribution by State', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Threshold Sensitivity - Precision
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(threshold_df['threshold'], threshold_df['precision'], 
                linewidth=3, color='#4ECDC4', label='Precision')
        ax3.axvline(x=0.16, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Current Threshold')
        optimal_idx = threshold_df['precision'].idxmax()
        ax3.scatter(threshold_df.loc[optimal_idx, 'threshold'], 
                   threshold_df.loc[optimal_idx, 'precision'],
                   s=200, c='red', marker='*', zorder=5, label='Optimal')
        ax3.set_xlabel('EAR Threshold', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 4. Threshold Sensitivity - All Metrics
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(threshold_df['threshold'], threshold_df['precision'], 
                linewidth=2, label='Precision', color='#4ECDC4')
        ax4.plot(threshold_df['threshold'], threshold_df['recall'], 
                linewidth=2, label='Recall', color='#FF6B6B')
        ax4.plot(threshold_df['threshold'], threshold_df['f1_score'], 
                linewidth=2, label='F1-Score', color='#95E1D3')
        ax4.axvline(x=0.16, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax4.set_xlabel('EAR Threshold', fontweight='bold')
        ax4.set_ylabel('Score', fontweight='bold')
        ax4.set_title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # 5. Frame Count Impact
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(frame_df['seconds'], frame_df['precision'], 
                linewidth=3, color='#4ECDC4', marker='o', markersize=8)
        ax5.axvline(x=3.0, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Current (3s)')
        ax5.set_xlabel('Detection Window (seconds)', fontweight='bold')
        ax5.set_ylabel('Precision', fontweight='bold')
        ax5.set_title('Detection Window Impact on Precision', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # 6. False Alarms vs Frame Count
        ax6 = plt.subplot(3, 3, 6)
        ax6.bar(frame_df['seconds'], frame_df['false_alarms'], 
               color='#FF6B6B', edgecolor='black', alpha=0.7)
        ax6.axvline(x=3.0, color='blue', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Current (3s)')
        ax6.set_xlabel('Detection Window (seconds)', fontweight='bold')
        ax6.set_ylabel('False Alarms', fontweight='bold')
        ax6.set_title('False Alarms vs Detection Window', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7. State Pie Chart
        ax7 = plt.subplot(3, 3, 7)
        state_counts = pd.Series(ground_truth).value_counts()
        colors_pie = ['#4ECDC4', '#FF6B6B']
        ax7.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax7.set_title('Ground Truth State Distribution', fontsize=14, fontweight='bold')
        
        # 8. Precision vs Recall Trade-off
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(threshold_df['recall'], threshold_df['precision'], 
                linewidth=3, color='#4ECDC4')
        ax8.scatter(threshold_df['recall'], threshold_df['precision'], 
                   c=threshold_df['threshold'], cmap='viridis', s=50, alpha=0.6)
        ax8.set_xlabel('Recall', fontweight='bold')
        ax8.set_ylabel('Precision', fontweight='bold')
        ax8.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax8.grid(alpha=0.3)
        cbar = plt.colorbar(ax8.collections[0], ax=ax8)
        cbar.set_label('Threshold', fontweight='bold')
        
        # 9. ROC-like Curve
        ax9 = plt.subplot(3, 3, 9)
        fpr = threshold_df['false_positives'] / len(ear_data)
        tpr = threshold_df['recall']
        ax9.plot(fpr, tpr, linewidth=3, color='#4ECDC4')
        ax9.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random')
        ax9.set_xlabel('False Positive Rate', fontweight='bold')
        ax9.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
        ax9.set_title('ROC-like Curve', fontsize=14, fontweight='bold')
        ax9.legend()
        ax9.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/detection_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Analysis visualization saved to: detection_analysis.png")
        
        return fig

def main():
    """Run comprehensive testing and analysis"""
    
    print("="*80)
    print("DRIVER DROWSINESS DETECTION - TESTING & CALIBRATION")
    print("="*80)
    
    tester = DetectionTester()
    
    # Generate test data (simulating 5,000+ frames)
    print("\n[1] Generating test data (5,000+ frames)...")
    ear_data, ground_truth = tester.simulate_ear_sequence(duration_seconds=180)  # 3 minutes = 5,400 frames at 30 FPS
    print(f"✓ Generated {len(ear_data)} frames")
    print(f"  • Alert frames: {ground_truth.count('alert')}")
    print(f"  • Drowsy frames: {ground_truth.count('drowsy')}")
    
    # Test threshold sensitivity
    print("\n[2] Testing threshold sensitivity...")
    threshold_results = tester.test_threshold_sensitivity(ear_data, ground_truth)
    
    optimal_threshold = threshold_results.loc[threshold_results['precision'].idxmax(), 'threshold']
    optimal_precision = threshold_results['precision'].max()
    
    print(f"✓ Tested 50 different thresholds")
    print(f"  • Optimal threshold: {optimal_threshold:.3f}")
    print(f"  • Maximum precision: {optimal_precision*100:.1f}%")
    print(f"  • Current threshold (0.16) precision: {threshold_results[threshold_results['threshold'].round(2) == 0.16]['precision'].values[0]*100:.1f}%")
    
    # Test frame count impact
    print("\n[3] Analyzing detection window impact...")
    frame_results = tester.analyze_frame_count_impact(ear_data, ground_truth)
    
    baseline_fa = frame_results[frame_results['seconds'] == 3.0]['false_alarms'].values[0]
    best_fa = frame_results['false_alarms'].min()
    reduction = ((baseline_fa - best_fa) / baseline_fa) * 100 if baseline_fa > 0 else 0
    
    print(f"✓ Tested detection windows from 1 to 6 seconds")
    print(f"  • 3-second window false alarms: {baseline_fa}")
    print(f"  • Minimum false alarms: {best_fa}")
    print(f"  • False alarm reduction: {reduction:.1f}%")
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    tester.visualize_results(ear_data, ground_truth, threshold_results, frame_results)
    
    # Save results
    print("\n[5] Saving test results...")
    threshold_results.to_csv('/home/claude/threshold_analysis.csv', index=False)
    frame_results.to_csv('/home/claude/frame_analysis.csv', index=False)
    print("✓ Results saved to CSV files")
    
    # Summary statistics
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\nDataset Statistics:")
    print(f"  • Total frames analyzed: {len(ear_data)}")
    print(f"  • Duration: {len(ear_data) / 30:.1f} seconds")
    print(f"  • Mean EAR: {np.mean(ear_data):.3f}")
    print(f"  • Std EAR: {np.std(ear_data):.3f}")
    
    print("\nOptimal Configuration:")
    print(f"  • Threshold: {optimal_threshold:.3f}")
    print(f"  • Detection Window: 3.0 seconds (90 frames)")
    print(f"  • Expected Precision: {optimal_precision*100:.1f}%")
    
    print("\nPerformance Metrics (at optimal threshold):")
    optimal_row = threshold_results.loc[threshold_results['precision'].idxmax()]
    print(f"  • Precision: {optimal_row['precision']*100:.1f}%")
    print(f"  • Recall: {optimal_row['recall']*100:.1f}%")
    print(f"  • F1-Score: {optimal_row['f1_score']*100:.1f}%")
    print(f"  • Accuracy: {optimal_row['accuracy']*100:.1f}%")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
