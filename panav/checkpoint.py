
import os

def get_latest_checkpoint(checkpoint_dir):
    ckpts = os.listdir(checkpoint_dir)
    max_iter = -1
    latest_checkpoint = ""
    for ck in ckpts:
        s = ck.split('_')
        # print(int(s[1]))
        if int(s[1])>max_iter:
            max_iter = int(s[1])
            latest_checkpoint = ck
    return latest_checkpoint