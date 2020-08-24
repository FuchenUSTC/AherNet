import argparse
import numpy as np
import sys
sys.path.append('./evaluation')
import matplotlib.pyplot as plt
import json

from eval_proposal import ANETproposal

def main(ground_truth_filename, proposal_filename, max_avg_nr_proposals=100,
         tiou_thresholds=np.linspace(0.5, 0.95, 10),
         subset='validation', verbose=True, check_status=True):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=True)
    anet_proposal.evaluate()

def run_evaluation(ground_truth_filename, proposal_filename, 
                   max_avg_nr_proposals=100, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=True)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    
    return (average_nr_proposals, average_recall, recall)

def plot_metric(average_nr_proposals, average_recall, recall, 
                tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    fn_size = 14
    plt.figure(num=None, figsize=(6, 5))
    ax = plt.subplot(1,1,1)
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)

    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)#

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)

    plt.show()

def plot_figure(ground_truth_filename, proposal_filename, max_avg_nr_proposals=100,
         tiou_thresholds=np.linspace(0.5, 0.95, 10),
         subset='validation', verbose=True, check_status=True):
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
    ground_truth_filename,
    proposal_filename,
    max_avg_nr_proposals=100,
    tiou_thresholds=np.linspace(0.5, 0.95, 10),
    subset='validation')
    plot_metric(uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)      

def evaluate_return_area(ground_truth_filename,proposal_filename,max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5,0.95,10),
        subset='validation',verbose = True, check_status = True):
    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=True)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video

    area_under_curve = np.trapz(average_recall, average_nr_proposals)
    AR_AN = 100.*float(area_under_curve)/average_nr_proposals[-1]
    Recall_all = 100.*float(average_recall[-1])
    
    return (AR_AN,Recall_all)         

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'proposal task which is intended to evaluate the ability '
                   'of algorithms to generate activity proposals that temporally '
                   'localize activities in untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('proposal_filename',
                   help='Full path to json file containing the proposals.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

def write_ar_an(gt, json_path, out_file):
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        gt,json_path,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')
    with open(out_file,'w') as fw:
        for k in range(len(uniform_average_nr_proposals_valid)):
            fw.write('%f %f\n'%(uniform_average_nr_proposals_valid[k],uniform_average_recall_valid[k]))
