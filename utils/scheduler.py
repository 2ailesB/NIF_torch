def scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    elif epoch < 2000:
        return 1e-3
    elif epoch < 4000:
        return 5e-4
    else:
        return 1e-4
