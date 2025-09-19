import matplotlib.pyplot as plt
import numpy as np

def plot_radar(name, scores, save_path=None):
    labels = list(scores.keys())
    values = list(scores.values()) + [scores[labels[0]]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"Stability Radar: {name}")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_dashboard(name, res):
    """
    Generate a comprehensive dashboard with text summary and radar plot.
    
    Parameters:
    -----------
    name : str
        Name of the model/dataset for display
    res : dict
        Results dictionary containing 'baseline', 'scores', and 'stability_overall'
        
    Returns:
    --------
    None
        Prints summary to console and displays radar plot
    """
    print(f"\n📊 {name} Stability Summary")
    print(f"Baseline: {res['baseline']:.4f}")
    for k, v in res["scores"].items():
        print(f"{k.capitalize()} Score: {v:.3f}")
    print(f"Overall Stability Score: {res['stability_overall']:.3f}")
    plot_radar(name, res["scores"])
