def nifs_scheduler5000(epoch, lr):
    if epoch < 1000:
        return lr
    elif epoch < 2000:
        return 1e-3
    elif epoch < 4000:
        return 5e-4
    else:
        return 1e-4

def nifms_scheduler5000(epoch, lr):
        if epoch < 1000:
            return lr
        elif epoch < 2000:
            return 1e-4
        elif epoch < 4000:
            return 5e-5
        else:
            return 1e-5

def nifll_scheduler600(epoch, lr):
        if epoch < 200:
            return lr
        elif epoch < 400:
            return 5e-4
        elif epoch < 600:
            return 2e-4
        else:
            return 3e-5

def nif_stepscheduler(epoch, lr):
        if epoch < 1000:
            return lr
        elif epoch < 1500:
            return 1e-4
        elif epoch < 2000:
            return 8e-5
        elif epoch < 2500:
            return 5e-5
        elif epoch < 3000:
            return 1e-5
        elif epoch < 3500:
            return 8e-6
        elif epoch < 4000:
            return 5e-6
        elif epoch < 4500:
            return 1e-6
        else :
            return 1e-7
            