class Factory:
    def __init__(self):
        self.creator = {}

    def register(self, name, creator):
        self.creator[name] = creator

    def create(self, name, *args, **kwargs):
        if name in self.creator:
            return self.creator[name](*args, **kwargs)
        else:
            raise ValueError(f"Creator with name {name} not found")


