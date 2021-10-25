import matplotlib.pyplot as plt
from filesys_manager import ExperimentPath
import time
import numpy as np

def plot_all(path, stopt):
    exp_path = ExperimentPath(path)
    # while True:
        # time.sleep(5)
    csvs = [ExperimentPath(f'{path}/run{run}/stats').csv_read(nrows=None) for run in range(10)]
    current_t = [m.shape[0] for m in csvs]
    print(current_t)
    for i, csv in enumerate(csvs):
        csvs[i] = csv[:min(current_t)]

    plt.title('squared bellman loss')
    plt.yscale('log')
    for csv in csvs:
        plt.plot(csv[:,0], csv[:,1], label='single', color='blue')
        plt.plot(csv[:,0], csv[:,2], label='ensemble', color='green')
    # plt.legend()
    plt.savefig(exp_path.squared_bellman_loss._path)
    plt.close()

    plt.title('squared bellman loss overall')
    # plt.yscale('log')
    single = []
    ensemble = []
    for csv in csvs:
        single.append(csv[:, 1])
        ensemble.append(csv[:, 2])
    single = np.stack(single).astype(float)
    ensemble = np.stack(ensemble).astype(float)
    single_mean = np.nanmean(single, axis=0)
    single_std = np.nanstd(single, axis=0)
    plt.plot(csv[:, 0], single_mean, label='single', color='blue')
    plt.fill_between(csv[:, 0], single_mean + single_std, single_mean - single_std, color='blue', alpha=.2)
    ensemble_mean = np.nanmean(ensemble, axis=0)
    ensemble_std = np.nanstd(ensemble, axis=0)
    plt.plot(csv[:, 0], ensemble_mean, label='ensemble', color='green')
    plt.fill_between(csv[:, 0], ensemble_mean + ensemble_std, ensemble_mean - ensemble_std, color='green', alpha=.2)
    plt.legend()
    plt.savefig(exp_path.squared_bellman_loss_overall._path)
    plt.close()

    plt.title('ground truth mse')
    plt.yscale('log')
    for csv in csvs:
        plt.plot(csv[:,0], csv[:,3], label='single', color='blue')
        plt.plot(csv[:,0], csv[:,4], label='ensemble', color='green')
    # plt.legend()
    plt.savefig(exp_path.ground_truth_mse._path)
    plt.close()

    plt.title('ground truth mse overall')
    plt.yscale('log')
    single = []
    ensemble = []
    for csv in csvs:
        single.append(csv[:, 3])
        ensemble.append(csv[:, 4])
    single = np.stack(single).astype(float)
    ensemble = np.stack(ensemble).astype(float)
    single_mean = np.nanmean(single, axis=0)
    single_std = np.nanstd(single, axis=0)
    plt.plot(csv[:, 0], single_mean, label='single', color='blue')
    plt.fill_between(csv[:, 0], single_mean + single_std, single_mean - single_std, color='blue', alpha=.2)
    ensemble_mean = np.nanmean(ensemble, axis=0)
    ensemble_std = np.nanstd(ensemble, axis=0)
    plt.plot(csv[:, 0], ensemble_mean, label='ensemble', color='green')
    plt.fill_between(csv[:, 0], ensemble_mean + ensemble_std, ensemble_mean - ensemble_std, color='green', alpha=.2)
    plt.legend()
    plt.savefig(exp_path.ground_truth_mse_overall._path)
    plt.close()

    for t in current_t:
        if t != stopt:
            break
    else:
        return


if __name__ == '__main__':
    # plot_all('exp/exp1', stopt=100_000)
    # plot_all('exp/exp1_replay', stopt=100_000)
    # plot_all('exp/exp2', stopt=100_000)
    plot_all('exp/exp2_replay', stopt=100_000)
    plot_all('exp/exp3_linear', stopt=100_000)
    plot_all('exp/exp3_linear_replay', stopt=100_000)
    plot_all('exp/exp3_relu', stopt=100_000)
    plot_all('exp/exp3_relu_replay', stopt=100_000)