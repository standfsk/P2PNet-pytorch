import os
import matplotlib.pyplot as plt

input_paths = os.listdir('ckpt')
save_path = os.path.join('graph')
os.makedirs(save_path, exist_ok=True)

train_by_epoch = dict()
test_by_epoch1 = dict()
test_by_epoch2 = dict()

for input_path in input_paths:
    file_path = os.path.join('ckpt', input_path, 'run_log.txt')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        counter = 0
        for line_id, line in enumerate(lines):
            if 'loss/loss@' in line:
                info = line.split("loss/loss@")[-1]
                epoch = info.split(":")[0]
                loss = info.split(":")[1][1:].split(' ')[0]
                train_by_epoch[int(epoch)] = round(float(loss), 5)
                counter = int(epoch)
            elif 'metric/mae@' in line:
                info = line.split("metric/mae@")[-1]
                epoch = counter
                mse = info.split(":")[2][1:].split(' ')[0]
                mae = info.split(":")[1][1:].split(' ')[0]
                test_by_epoch1[int(epoch)] = round(float(mae), 5)
                test_by_epoch2[int(epoch)] = round(float(mse), 5)
    
    plt.clf()
    plt.plot(train_by_epoch.keys(), train_by_epoch.values())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.savefig(os.path.join(save_path, f'{input_path}_train.jpg'))

    plt.clf()
    plt.plot(test_by_epoch1.keys(), test_by_epoch1.values(), label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Val MAE')
    plt.savefig(os.path.join(save_path, f'{input_path}_val.jpg'))