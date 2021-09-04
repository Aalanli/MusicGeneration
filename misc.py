class EasyDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"{name} not in dictionary")
    def __setattr__(self, name: str, value) -> None:
        self[name] = value 
    def search_common_naming(self, name, seperator='_'):
        name = name + seperator
        return {k.replace(name, ''): v for k, v in self.items() if name in k}
    def get_copy(self):
        return EasyDict(self.copy())