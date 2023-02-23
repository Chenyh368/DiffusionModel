class ScalerMeter(object):

    def __init__(self):
        self.x = None

    def update(self, x):
        if not isinstance(x, (int, float)):
            x = x.item()
        self.x = x

    def reset(self):
        self.x = None

    def get_value(self):
        if self.x:
            return self.x
        return 0


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, x, n=1):
        self.sum += float(x)
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def get_value(self):
        if self.n:
            return self.sum / self.n
        return 0