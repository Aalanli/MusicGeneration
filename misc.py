import math


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


class CustomScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, param_groups=(0,)) -> None:
        self.opt = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.param_groups = param_groups
        self.const = 1 / math.sqrt(self.d_model)
    
    def step(self, step):
        step += 1
        arg1 = 1 / math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        for p in self.param_groups:
            self.opt.param_groups[p]['lr'] = self.const * min(arg1, arg2)
        