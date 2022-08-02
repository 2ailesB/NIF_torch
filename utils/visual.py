import matplotlib.pyplot as plt
import numpy as np

def visual_1dwave(model, datax, datay, path, mode):
    nsamples = datax.shape[0]
    xx = datax[:, 0].reshape(int(nsamples/200), 200).to('cpu')
    tt = datax[:, 1].reshape(int(nsamples/200), 200).to('cpu')
    # xx, tt = np.meshgrid(x, t)
    # print("xx.shape :", xx.shape)
    # print("datax.shape :", datax.shape)
    # print("model(datax).shape :", model(datax).shape)

    u_pred = model(
    datax).reshape(int(nsamples/200), 200).detach().to('cpu')
    datay = datay.reshape(int(nsamples/200), 200)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    im1 = axs[0].contourf(tt, xx, datay.reshape(
        int(nsamples/200), 200), vmin=-5, vmax=5, levels=50, cmap='seismic')
    plt.colorbar(im1, ax=axs[0])
    im2 = axs[1].contourf(tt, xx,
                            u_pred, vmin=-5, vmax=5, levels=50, cmap='seismic')
    plt.colorbar(im2, ax=axs[1])
    # print("u_pred.shape :", u_pred.shape)
    # print("datay.shape :", datay.shape)
    im3 = axs[2].contourf(tt, xx, (
        u_pred - datay).reshape(int(nsamples/200), 200), vmin=-5, vmax=5, levels=50, cmap='seismic')
    plt.colorbar(im3, ax=axs[2])
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')
    axs[0].set_title('true')
    axs[1].set_title('pred')
    axs[2].set_title('error')
    plt.savefig(path + f'/vis_{mode}.png')
    plt.close()
    plt.clf()
    return True