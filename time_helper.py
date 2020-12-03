import time
from functools import wraps


def log_time_cost(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        time1 = time.time()
        results = f(*args, **kwds)
        time2 = time.time()
        print({
            "log_type": "FUNC_TIME_COST",
            "function": f.__name__,
            "cost_sec": (time2 - time1)
        })
        return results
    return wrapper
