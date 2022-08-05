import matplotlib.pyplot as plt
import numpy as np
import torch

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

def visual_cylinder(model, datax, datay, path, mode):

    index_ = np.diff(datax[:, 0].cpu().detach().numpy(), append=0) > 0
    time_jump_index = np.arange(datax.shape[0])[index_]+1
    datax = datax[-time_jump_index[-1]:, :]
    datay = datay[-time_jump_index[-1]:, :]
    
    data = torch.cat((datax, datay), dim=1)
    # stds = data.std(0).to('cpu').detach().numpy()
    # means = data.mean(0).to('cpu').detach().numpy()
    stds = np.array([ 1.0821e-01,  1.4988e-02,  5.6368e-05,  2.8442e+00, -2.7569e-02])
    means = np.array([0.0029, 0.0139, 0.0093, 1.1646, 1.3246])

    uv_pred = model(datax[:, 0:3]).to('cpu').detach().numpy()  # (300, 2)
    u_pred = uv_pred[:, 0] * stds[3] + means[3]
    v_pred = uv_pred[:, 1] * stds[4] + means[4]
    datax = datax.to('cpu').detach().numpy()
    datay = datay.to('cpu').detach().numpy()
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    fig, axs = plt.subplots(2, 3, figsize=(16, 4))
    plt.set_cmap('PRGn')
    im1 = axs[0, 0].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2],
                                datay[:, 0] * stds[3] + means[3], vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    axs[0, 0].add_patch(circle)
    plt.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2],
                                u_pred, vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    axs[0, 1].add_patch(circle)
    plt.colorbar(im2, ax=axs[0, 1])
    im3 = axs[0, 2].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2], (
        (u_pred) - (datay[:, 0] * stds[3] + means[3])), vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    axs[0, 2].add_patch(circle)
    plt.colorbar(im3, ax=axs[0, 2])
    im1 = axs[1, 0].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2],
                                datay[:, 1] * stds[4] + means[4], vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    axs[1, 0].add_patch(circle)
    plt.colorbar(im1, ax=axs[1, 0])
    im2 = axs[1, 1].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2],
                                v_pred, vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    axs[1, 1].add_patch(circle)
    plt.colorbar(im2, ax=axs[1, 1])
    im3 = axs[1, 2].tricontourf(datax[:, 1].flatten() * stds[1] + means[1], datax[:, 2].flatten() * stds[2] + means[2], (
        (v_pred) - (datay[:, 1] * stds[4] + means[4])), vmin=-5, vmax=5, levels=50)  # , cmap='seismic')
    circle = plt.Circle((0, 0), 0.0035, color='grey')
    axs[1, 2].add_patch(circle)
    plt.colorbar(im3, ax=axs[1, 2])
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title('u true')
    axs[0, 1].set_title('u pred')
    axs[0, 2].set_title('u error')
    axs[1, 0].set_title('v true')
    axs[1, 1].set_title('v pred')
    axs[1, 2].set_title('v error')
    plt.savefig(path + f'/vis_{mode}.png')
    plt.close()
    plt.clf()

    return True