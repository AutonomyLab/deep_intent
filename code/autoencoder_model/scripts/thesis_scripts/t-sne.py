import time
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
# from ggplot import *


def build_tsne(z, labels):
    # PCA plot
    pca = PCA(n_components=100)
    pca_result = pca.fit_transform(z)
    print(pca_result.shape)

    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(pca_result)

    # print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
    print (tsne_results.shape)
    a1 = tsne_results[:, 0]
    a2 = tsne_results[:, 1]
    a3 = tsne_results[:, 2]

    np.save('/local_home/JAAD_Dataset/thesis/t-sne.npy', tsne_results)

    fig = plt.figure()
    ax = fig.add_subplot(111, title='t-sne', projection='3d')
    # Create the scatter
    mesh = ax.scatter(xs=a1, ys=a2, zs=a3, c=labels, edgecolors='none')
    plt.colorbar(mesh, ax=ax)
    plt.grid()
    # ax.colorbar()
    plt.show()

if __name__ == "__main__":
    z = np.load('/local_home/JAAD_Dataset/thesis/results/res/test_results/graphs/values/z_all.npy')
    print (z.shape)
    z = np.reshape(z, newshape=(1257, 16, 16, 26, 64))
    print (z.shape)
    z = np.moveaxis(z, 0, 1)
    print (z.shape)
    z_flat = []
    z_pca = []
    for i in range(1, 16):
        z_sub_flat = []
        for j in range (1257):
            z_pca.append(z[i, j].flatten() - z[0, j].flatten())
            z_sub_flat.append(z[i, j].flatten())
        z_flat.append(z_sub_flat)
    z_pca = np.asarray(z_pca)
    print (z_pca)
    labels = np.asarray([2]*1257 + [3]*1257 + [4] *1257 + [5] *1257 + [6] *1257 + [7] *1257 + [8] *1257\
             + [9] * 1257 + [10] *1257 + [11] *1257 + [12] *1257 + [13] *1257 + [14] *1257 + [15] *1257\
             + [16] *1257)
    print (labels.shape)
    # 74FCFE #CE00C1 #965DFF #3AC500

    z_pca = np.column_stack((z_pca, labels))
    print (z_pca.shape)
    # Create random permutation to select 10k samples
    z_pca = np.random.permutation(z_pca)[0:10000]
    labels = z_pca[:, -1]
    print (labels.shape)
    z_pca = z_pca[:, 0:26624]
    print (z_pca.shape)
    print (labels)

    label_2 = []
    for label in labels:
        if label <=4:
            label_2.append('r')
        elif label <=8:
            label_2.append('w')
        elif label <=12:
            label_2.append('w')
        else:
            label_2.append('y')
    label_2 = np.asarray(label_2)
    print (label_2)

    np.save('/local_home/JAAD_Dataset/thesis/labels.npy', labels)
    build_tsne(z_pca, label_2)
