from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
np.random.seed(2**10)

def build_tsne(z, labels):
    # PCA plot
    # perplexity = 350
    # pca = PCA(n_components=100)
    # pca_result = pca.fit_transform(z)
    #
    # tsne3 = TSNE(n_components=3, verbose=2, perplexity=perplexity, n_iter=2000)
    # tsne3_results = tsne3.fit_transform(pca_result)
    #
    # tsne2 = TSNE(n_components=2, verbose=2, perplexity=perplexity, n_iter=2000)
    # tsne2_results = tsne2.fit_transform(pca_result)
    #
    # np.save('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec3-enc-' + str(perplexity) + '.npy', tsne3_results)
    # np.save('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec2-enc-' + str(perplexity) + '.npy', tsne2_results)
    # np.save('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/labels-rendec-enc-' + str(perplexity) + '.npy', labels)

    tsne3_results = np.load('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec3-lstm-100.npy')
    tsne2_results = np.load('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/t-sne-rendec2-lstm-100.npy')
    labels = np.load('/local_home/JAAD_Dataset/thesis/plots/data/t-sne/labels-rendec-lstm-100.npy')

    a1 = tsne3_results[:, 0]
    a2 = tsne3_results[:, 1]
    a3 = tsne3_results[:, 2]

    b1 = tsne2_results[:, 0]
    b2 = tsne2_results[:, 1]

    # Setup plot appearances
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                  'monospace': ['Computer Modern']})

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.5
    # plt.rcParams['axes.color'] = 'white'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.color'] = [0.8, 0.8, 0.8]

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111)
    # Create the scatter
    mesh1 = ax1.scatter(xs=a1, ys=a2, zs=a3, c=labels,
                      cmap='jet',
                      edgecolors='none')
    mesh2 = ax2.scatter(x=b1, y=b2, c=labels, cmap='jet', edgecolors='none')
    # plt.colorbar(mesh, ax=ax)
    plt.grid()

    plt.colorbar(mesh1, fraction=0.046, pad=0.04, ax=ax1)
    plt.colorbar(mesh2, fraction=0.046, pad=0.04, ax=ax2)

    ax1.w_xaxis.line.set_color('white')
    ax1.w_yaxis.line.set_color('white')
    ax1.w_zaxis.line.set_color('white')

    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')

    ax1.w_xaxis.set_pane_color((1, 1, 1, 1))
    ax1.w_yaxis.set_pane_color((1, 1, 1, 1))
    ax1.w_zaxis.set_pane_color((1, 1, 1, 1))

    ax2.set_facecolor([1, 1, 1])

    # plt.savefig('/local_home/JAAD_Dataset/thesis/plots/' + 'tsne-rendec2-lstm.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    z = np.load('/local_home/JAAD_Dataset/thesis/plots/data/z/z_all_rendec_enc.npy')
    # Remove the 1 dimensional second axis
    z = np.reshape(z, newshape=(1257, 16, 16, 26, 64))
    z_mean = np.mean(z, axis=1)
    z = np.moveaxis(z, 0, 1)
    z_pca = []
    n_samples = 625
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

