from argparse import ArgumentParser

class BaseCommand:
    arguments = list()
    paths = list()
    def __init__(self, inpute):
        parser = ArgumentParser()
        for arg in self.arguments:
            parser.add_argument(arg[0], arg[1], help = arg[2])
        self.args = parser.parse_args(inpute)
        
        for pth in self.paths:
            valp = getattr(self.args, pth)
            valp = self.pathify(valp)
            setattr(self.args, pth, valp)

    def add_arg(self, flag: str, name: str, comment: str, path=False):
        self.arguments.append((f'-{flag}',f'--{name}',comment))
        if path:
            self.paths.append(name)

    def submit(self):
        raise NotImplementedError

    @staticmethod
    def pathify(pth: str):
        if not pth.endswith('/'):
            pth += '/'
        return pth

