import time

class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time()
    def stop(self, label=""): 
        dt = time.time() - self.t0
        print(f"{label}: {dt:.3f}s"); return dt
