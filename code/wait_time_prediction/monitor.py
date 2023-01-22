
"""Copyright (c) Microsoft Corporation.

Taken from Project OLive (Microsoft) with modifications.
"""

import torch
from threading import Thread
import time
import numpy as np


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gpu_usage = []
        self.start_time = time.time()
        self.end_time = None
        self.start()

    def run(self):
        while not self.stopped:
            if self.device == 'cuda:0':
                self.gpu_usage.append(torch.cuda.utilization(self.device))
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        self.end_time = time.time()
        
    def get_average_gpu_usage_and_time_elapsed(self):
        return np.mean(self.gpu_usage), self.end_time - self.start_time
    