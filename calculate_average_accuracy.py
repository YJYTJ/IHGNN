import argparse
import numpy as np

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", help="dataset name")
        args = parser.parse_args()

        epochs = 350
        layers = ['2','4']
        batches = ['32','64','128']
        learning_rates = [0.01, 0.001, 0.0001]
        result_acc = []
        result_std = []
        for layer in layers:
            for batch in batches:
                for learning_rate in learning_rates:

                    results_folder = 'results/result{}_{}_{}'.format(layer, batch, learning_rate)
                    validation_loss = np.zeros((epochs, 10))
                    test_accuracy = np.zeros((epochs, 10))
                    test_acc = np.zeros(10)

                    with open(results_folder+'/{}_acc_results.txt'.format(args.dataset), 'r') as filehandle:
                        filecontents = filehandle.readlines()
                        index = 0
                        col = 0
                        for line in filecontents:
                            ss = line.split()
                            t_acc = ss[-1]
                            v_loss = ss[-2]
                            validation_loss[index][col] = float(v_loss)
                            test_accuracy[index][col] = float(t_acc)
                            index += 1
                            if index == epochs:
                                index = 0
                                col += 1
                                if col == 10:
                                    break

                    min_ind = np.argmin(validation_loss, axis=0)
                    for i in range(10):
                        ind = min_ind[i]
                        test_acc[i] = test_accuracy[ind][i]
                    ave_acc = np.mean(test_acc)
                    std_acc = np.std(test_acc)

                    result_acc.append(ave_acc)
                    result_std.append(std_acc)
        best_acc = max(result_acc)
        idx = np.argmax(result_acc)
        best_std = result_std[idx]
        print('test accuracy / mean(std): {0:.5f}({1:.5f})'.format(best_acc, best_std))

    except IOError as e:
        print(e)