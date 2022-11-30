import sys

class Boss:
    def __init__(self):
        self.els = dict()

    def add_command(self, key: str, func: callable):
        self.els[key] = func

    def get_command(self, key):
        return self.els[key]()

def get_stats():
        from commands.get_stats import GetStats as comandor
        return comandor

def get_out_stats():
        from commands.get_stats import GetOutStats as comandor
        return comandor

if __name__=='__main__':
    boss = Boss()
    boss.add_command('get_stats', get_stats)
    boss.add_command('get_out_stats', get_out_stats)
    
    
    try:
        subcommand = sys.argv[1]

        Comador = boss.get_command(subcommand) 
        task = Comador(sys.argv[2:])
        task.submit()
    except IndexError:
        print('Options: ')
        print(list(boss.els.keys()))  # Display help if no arguments were given.
