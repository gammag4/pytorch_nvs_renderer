import time
import torch
from easydict import EasyDict as edict


_should_profile = False
_regions = {}
_warmup = 0
_curr_iter = 0


def start(*, warmup=0):
    global _regions
    global _should_profile
    global _warmup
    global _curr_iter

    _regions = {}
    _warmup = warmup
    _curr_iter = 0
    _should_profile = True


def stop():
    global _should_profile

    _should_profile = False


def step():
    global _curr_iter

    _curr_iter += 1


class RegionProfiler:
    def __init__(self, region_name):
        global _should_profile
        global _regions
        global _warmup
        global _curr_iter

        if not _should_profile or _curr_iter < _warmup:
            self.profiling = False
            return

        self.profiling = True
        self.region_name = region_name
        self.region_data = _regions.get(region_name, [])
        _regions[region_name] = self.region_data
        self.start = None

    def __enter__(self):
        if self.profiling:
            self.start = time.perf_counter()

        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiling:
            end = time.perf_counter()
            self.region_data.append((self.start, end, end - self.start))

        if exc_type is not None:
            return False
        return True


class Profiler:
    def __init__(self, *, should_profile=True, warmup=0):
        self.should_profile = should_profile
        self.warmup = warmup

    def __enter__(self):
        if self.should_profile:
            start(warmup=self.warmup)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.should_profile:
            stop()

        if exc_type is not None:
            return False
        return True

    def step(self):
        step()


def human_readable(value, unit):
    if value == 0:
        return f' {0.0:06.2f}{unit}'

    units = [
        (1e15, 'P'), (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'), (1, ''),
        (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'), (1e-12, 'p'), (1e-15, 'f')
    ]

    s = None
    abs_val = abs(value)
    for f, u in units:
        if abs_val >= f:
            s = f'{value / f:06.2f}{u}'
            break

    if s is None:
        s = f'{value:06.2e}'

    return f'{s:>07}{unit}'


def get_results():
    global _regions

    results = _regions
    results = {k: torch.tensor(v)[:, 2] for k, v in results.items()}
    results = {k: tuple(i.item() for i in (v.mean(), v.std(), v.max(), v.min())) for k, v in results.items()} # TODO fix distribution
    results = {k: edict(mean=v[0], std=v[1], max=v[2], min=v[3]) for k, v in results.items()}
    return results


def print_results():
    results = get_results()
    max_len = max(len(i) for i in results.keys())
    
    print('Profiling results:')

    if not results:
        print('No profiling was conducted')
        return

    for k, v in results.items():
        s = ', '.join(f'{k2}={human_readable(v2, 's')}' for k2, v2 in v.items())  # TODO fix format
        sname = f'`{k}`:'
        print(f'{sname:<{max_len + 3}} {s}')


# Three cols: start, end, interval
def dump(fpath):
    global _regions

    results = _regions
    results = {k: torch.tensor(v) for k, v in results.items()}
    torch.save(results, fpath)
