import os
import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr, roc_auc, output_dir, logger, fig_save_name='roc_curve.png'):
    save_path = os.path.join(output_dir, fig_save_name)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    logger.info(f"Saved ROC curve plot to {save_path}")
    
    
def plot_pr_curve(recall, precision, pr_auc, avg_precision, output_dir, logger, fig_save_name='pr_curve.png'):
    save_path = os.path.join(output_dir, fig_save_name)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'PR curve (area = {pr_auc:.3f}, avg precision = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    logger.info(f"Saved PR curve plot to {save_path}")