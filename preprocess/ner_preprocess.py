
class DataLoader(object):
    def __init__(self):
        pass

    @classmethod
    def load_data(cls,file):
        with open(file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line.split()
