import os
from matplotlib import pyplot as plt



def cluster_output_handler(listResult):
    clusterIndexList = []
    for i in range(len(set(listResult))):
        clusterIndexList.append([])
    for i in range(len(listResult)):
        clusterIndexList[listResult[i]].append(i)

    return clusterIndexList

def edgeList2edgeIndex(edgeList):
    result=[[i[0],i[1]] for i in edgeList]
    return result

def plot(y, xlabel='epochs', ylabel='', hline=None, output_dir='', suffix=''):
    plt.plot(range(len(y)), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=hline, color='green', linestyle='-') if hline else None
    plt.savefig(os.path.join(output_dir, f"{ylabel.replace(' ', '_')}{suffix}.png"), dpi=200)
    plt.clf()

