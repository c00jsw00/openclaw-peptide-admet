#!/usr/bin/env python3
"""
Generate Graphical Abstract and TOC Graphic for JCIM Submission
================================================================

Creates high-resolution PNG images for manuscript submission.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_graphical_abstract(output_path='graphical_abstract.png'):
    """
    Create Graphical Abstract (4000x2400 px)
    """
    print("Creating Graphical Abstract...")
    
    # Set figure size (4000x2400 px at 100 DPI = 40x24 inches)
    fig, ax = plt.subplots(figsize=(40, 24), dpi=100)
    
    # Clear axes
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 24)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(20, 22, 'Ensemble ML Model for Peptide ADMET Property Prediction', 
            ha='center', va='center', fontsize=48, fontweight='bold', 
            family='Arial', color='#212121')
    
    # Section 1: Peptide Sequence (Left)
    # Background box
    ax.add_patch(plt.Rectangle((2, 10), 8, 10, fill=True, facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=3))
    
    # Title
    ax.text(6, 18, 'Peptide Sequence', ha='center', va='center', fontsize=24, fontweight='bold', 
            family='Arial', color='#1565C0')
    
    # Sequence
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    y_start = 16
    for i, aa in enumerate(sequence):
        x = 4 + (i % 10) * 0.6
        y = y_start - (i // 10) * 0.8
        color = '#2196F3' if aa in 'PR' else '#1976D2' if aa in 'GA' else '#1565C0'
        ax.text(x, y, aa, ha='center', va='center', fontsize=18, fontweight='bold', 
                family='Arial', color=color)
    
    # Statistics
    ax.text(6, 12, 'Length: 8-25 AA\nSize: 500-5000 Da', ha='center', va='center', fontsize=16, 
            family='Arial', color='#424242')
    
    # Section 2: Feature Engineering (Middle)
    # Background box
    ax.add_patch(plt.Rectangle((12, 10), 16, 10, fill=True, facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=3))
    
    # Title
    ax.text(20, 18, '428D Feature Engineering', ha='center', va='center', fontsize=24, fontweight='bold', 
            family='Arial', color='#E65100')
    
    # Feature breakdown
    features = [
        ('AAC', '20 features', '#FF9800'),
        ('DPC', '400 features', '#F57C00'),
        ('PhysChem', '8 features', '#EF6C00')
    ]
    
    y = 16
    for name, desc, color in features:
        ax.add_patch(plt.Rectangle((14, y-0.5), 12, 0.8, fill=True, facecolor=color, edgecolor='black', linewidth=1))
        ax.text(20, y, f'{name}: {desc}', ha='center', va='center', fontsize=18, fontweight='bold', 
                family='Arial', color='white')
        y -= 1.2
    
    # Arrow
    ax.arrow(10, 15, 2, 0, head_width=0.5, head_length=0.8, fc='#9E9E9E', ec='#9E9E9E', linewidth=3)
    
    # Section 3: Ensemble Model (Right)
    # Background box
    ax.add_patch(plt.Rectangle((30, 10), 8, 10, fill=True, facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=3))
    
    # Title
    ax.text(34, 18, 'Ensemble Model', ha='center', va='center', fontsize=24, fontweight='bold', 
            family='Arial', color='#2E7D32')
    
    # Model components
    models = [
        ('Random Forest', '100 trees', '#4CAF50'),
        ('Neural Network', '[128,64,32]', '#66BB6A'),
        ('Averaging', 'Integration', '#81C784')
    ]
    
    y = 16
    for name, desc, color in models:
        ax.add_patch(plt.Rectangle((31, y-0.5), 6, 0.8, fill=True, facecolor=color, edgecolor='black', linewidth=1))
        ax.text(34, y, f'{name}: {desc}', ha='center', va='center', fontsize=16, fontweight='bold', 
                family='Arial', color='white')
        y -= 1.2
    
    # Arrow
    ax.arrow(38, 15, 2, 0, head_width=0.5, head_length=0.8, fc='#9E9E9E', ec='#9E9E9E', linewidth=3)
    
    # Section 4: ADMET Results (Bottom)
    # Background box
    ax.add_patch(plt.Rectangle((2, 2), 36, 7, fill=True, facecolor='#F3E5F5', edgecolor='#9C27B0', linewidth=3))
    
    # Title
    ax.text(20, 8, 'ADMET Prediction Results', ha='center', va='center', fontsize=24, fontweight='bold', 
            family='Arial', color='#6A1B9A')
    
    # Endpoint data
    endpoints = [
        ('GI Absorption', 97.70, '#2196F3'),
        ('Caco-2', 98.91, '#2196F3'),
        ('BBB', 98.47, '#2196F3'),
        ('Ames', 97.27, '#FF5722'),
        ('hERG', 97.91, '#FF5722')
    ]
    
    x_start = 4
    width = 6.5
    bar_height = 0.8
    y_bar = 6
    
    for i, (name, accuracy, color) in enumerate(endpoints):
        x = x_start + i * (width + 0.5)
        
        # Bar background
        ax.add_patch(plt.Rectangle((x, y_bar - 0.4), width, 0.8, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1))
        
        # Progress bar
        progress_width = (accuracy / 100) * (width - 0.4)
        ax.add_patch(plt.Rectangle((x + 0.2, y_bar - 0.3), progress_width, 0.6, fill=True, facecolor=color, edgecolor='black', linewidth=1))
        
        # Text
        ax.text(x + width/2, y_bar, f'{name}', ha='center', va='center', fontsize=16, fontweight='bold', 
                family='Arial', color='#424242')
        ax.text(x + width/2, y_bar - 0.5, f'{accuracy:.2f}%', ha='center', va='center', fontsize=20, fontweight='bold', 
                family='Arial', color=color)
    
    # Overall statistics
    ax.text(20, 3, 'Overall: 97.70% Accuracy | AUC-ROC: 0.9987 | +64.87% vs GNN', 
            ha='center', va='center', fontsize=20, fontweight='bold', 
            family='Arial', color='#6A1B9A')
    
    # GitHub URL
    ax.text(20, 1.5, 'Available at: github.com/c00jsw00/openclaw-peptide-admet', 
            ha='center', va='center', fontsize=16, 
            family='Arial', color='#616161')
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Graphical Abstract saved to {output_path}")
    return output_path


def create_toc_graphic(output_path='toc_graphic.png'):
    """
    Create TOC Graphic (1100x400 px)
    """
    print("Creating TOC Graphic...")
    
    # Set figure size (1100x400 px at 100 DPI = 11x4 inches)
    fig, ax = plt.subplots(figsize=(11, 4), dpi=100)
    
    # Clear axes
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Section 1: Peptide
    ax.add_patch(plt.Rectangle((0.2, 0.8), 2, 2.4, fill=True, facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=2))
    ax.text(1.2, 2.5, 'Peptide', ha='center', va='center', fontsize=18, fontweight='bold', 
            family='Arial', color='#1565C0')
    ax.text(1.2, 1.8, 'Sequence', ha='center', va='center', fontsize=14, fontweight='bold', 
            family='Arial', color='#1565C0')
    ax.text(1.2, 1.2, '8-25 AA', ha='center', va='center', fontsize=12, 
            family='Arial', color='#424242')
    
    # Arrow
    ax.arrow(2.2, 2, 0.5, 0, head_width=0.1, head_length=0.2, fc='#9E9E9E', ec='#9E9E9E', linewidth=2)
    
    # Section 2: Features
    ax.add_patch(plt.Rectangle((3.2, 0.8), 2.5, 2.4, fill=True, facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2))
    ax.text(4.45, 2.5, '428D', ha='center', va='center', fontsize=18, fontweight='bold', 
            family='Arial', color='#E65100')
    ax.text(4.45, 1.8, 'Features', ha='center', va='center', fontsize=14, fontweight='bold', 
            family='Arial', color='#E65100')
    ax.text(4.45, 1.2, 'AAC+DPC+P-Chem', ha='center', va='center', fontsize=10, 
            family='Arial', color='#424242')
    
    # Arrow
    ax.arrow(5.7, 2, 0.5, 0, head_width=0.1, head_length=0.2, fc='#9E9E9E', ec='#9E9E9E', linewidth=2)
    
    # Section 3: Model
    ax.add_patch(plt.Rectangle((6.7, 0.8), 2, 2.4, fill=True, facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))
    ax.text(7.7, 2.5, 'Ensemble', ha='center', va='center', fontsize=18, fontweight='bold', 
            family='Arial', color='#2E7D32')
    ax.text(7.7, 1.8, 'Model', ha='center', va='center', fontsize=14, fontweight='bold', 
            family='Arial', color='#2E7D32')
    ax.text(7.7, 1.2, 'RF+NN', ha='center', va='center', fontsize=12, 
            family='Arial', color='#424242')
    
    # Arrow
    ax.arrow(8.7, 2, 0.5, 0, head_width=0.1, head_length=0.2, fc='#9E9E9E', ec='#9E9E9E', linewidth=2)
    
    # Section 4: Results
    ax.add_patch(plt.Rectangle((9.7, 0.8), 1.8, 2.4, fill=True, facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))
    ax.text(10.6, 2.5, '97.70%', ha='center', va='center', fontsize=20, fontweight='bold', 
            family='Arial', color='#2E7D32')
    ax.text(10.6, 1.8, 'Accuracy', ha='center', va='center', fontsize=14, fontweight='bold', 
            family='Arial', color='#2E7D32')
    ax.text(10.6, 1.2, 'AUC-ROC: 0.9987', ha='center', va='center', fontsize=10, 
            family='Arial', color='#424242')
    
    # Title at bottom
    ax.text(5.5, 0.4, 'Ensemble ML for Peptide ADMET Prediction', ha='center', va='center', fontsize=14, fontweight='bold', 
            family='Arial', color='#212121')
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ TOC Graphic saved to {output_path}")
    return output_path


def main():
    """Generate both graphics"""
    print("="*70)
    print("JCIM Graphical Abstract Generator")
    print("="*70)
    
    output_dir = Path('.')
    
    # Generate graphical abstract
    ga_path = create_graphical_abstract(output_dir / 'graphical_abstract.png')
    
    # Generate TOC graphic
    toc_path = create_toc_graphic(output_dir / 'toc_graphic.png')
    
    print("\n" + "="*70)
    print("✅ Graphics generation complete!")
    print(f"   Graphical Abstract: {ga_path} (4000x2400 px)")
    print(f"   TOC Graphic: {toc_path} (1100x400 px)")
    print("="*70)


if __name__ == '__main__':
    main()
