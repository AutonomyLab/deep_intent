from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
np.random.seed(2**5)

def build_tsne(z, labels):
    # PCA plot
    perplexity = 100
    pca = PCA(n_components=100)
    pca_result = pca.fit_transform(z)

    # tsne = TSNE(n_components=3, verbose=2, perplexity=perplexity, n_iter=1500)
    tsne = TSNE(n_components=2, verbose=2, perplexity=perplexity, n_iter=1500)
    tsne_results = tsne.fit_transform(pca_result)

    np.save('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec2-' + str(perplexity) + '.npy', tsne_results)
    np.save('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/labels-rendec2-' + str(perplexity) + '.npy', labels)

    # tsne_results = np.load('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec2.npy')
    # labels = np.load('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/labels_rendec2.npy')

    a1 = tsne_results[:, 0]
    a2 = tsne_results[:, 1]
    # a3 = tsne_results[:, 2]

    # Setup plot appearances
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                  'monospace': ['Computer Modern']})

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.5
    # plt.rcParams['axes.color'] = 'white'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.color'] = [0.8, 0.8, 0.8]

    fig = plt.figure()
    # ax = fig.add_subplot(111, title='t-sne', projection='3d')
    ax = fig.add_subplot(111, title='t-sne')
    # Create the scatter
    # mesh = ax.scatter(xs=a1, ys=a2, zs=a3, c=labels,
    #                   cmap='jet',
    #                   edgecolors='none')
    mesh = ax.scatter(x=a1, y=a2, c=labels, cmap='viridis', edgecolors='none')
    # plt.colorbar(mesh, ax=ax)
    plt.colorbar(mesh, fraction=0.046, pad=0.04, ax=ax)
    plt.grid()
    ax.set_facecolor([1, 1, 1])
    plt.title("Perplexity =" + str(perplexity))
    plt.show()


if __name__ == "__main__":
    z = np.load('/local_home/JAAD_Dataset/thesis/plots/data/z/z_all_rendec.npy')
    # Remove the 1 dimensional second axis
    z = np.reshape(z, newshape=(1257, 16, 16, 26, 64))
    z_mean = np.mean(z, axis=1)
    z = np.moveaxis(z, 0, 1)
    z_pca = []
    n_samples = 500
    for i in range(1, 16):
        for j in range(n_samples):
            z_pca.append(z[i, j].flatten() - z[0, j].flatten())
            # z_pca.append(z[i, j].flatten() - z_mean[j].flatten())

    z_pca = np.asarray(z_pca)

    # labels = np.asarray([1]*n_samples + [2]*n_samples + [3]*n_samples + [4] *n_samples + [5] *n_samples + [6] *n_samples + [7] *n_samples + [8] *n_samples\
    #          + [9] * n_samples + [10] *n_samples + [11] *n_samples + [12] *n_samples + [13] *n_samples + [14] *n_samples + [15] *n_samples\
    #          + [16] *n_samples)
    labels = np.asarray(
        [2] * n_samples + [3] * n_samples + [4] * n_samples + [5] * n_samples + [6] * n_samples +
        [7] * n_samples + [8] * n_samples + [9] * n_samples + [10] * n_samples + [11] * n_samples +
        [12] * n_samples + [13] * n_samples + [14] * n_samples + [15] * n_samples + [16] * n_samples)

    # 74FCFE #CE00C1 #965DFF #3AC500

    print (labels)
    z_pca = np.column_stack((z_pca, labels))
    # Create random permutation to select 10k samples
    z_pca = np.random.permutation(z_pca)
    labels = z_pca[:, -1]
    z_pca = z_pca[:, 0:26624]

    # label_2 = []
    # for label in labels:
    #     if label <=4:
    #         label_2.append('r')
    #     elif label <=8:
    #         label_2.append('w')
    #     elif label <=12:
    #         label_2.append('w')
    #     else:
    #         label_2.append('y')
    # label_2 = np.asarray(label_2)
    # print (label_2)

    print (labels)

    build_tsne(z_pca, labels)

