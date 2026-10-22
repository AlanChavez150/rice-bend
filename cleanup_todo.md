# mgs.py

1. Completely remove extremely long and excessive comments
2. Remove all documentation about how the code used to be. only keep comments that describe how the code behaves now
3. Change imports to be more concise. For example:

Instead of:
```
from rice_bend.data_store import GSHistory, check_run_dir, make_run_dir, save_run
```

Do:
```
from rice_bend import data_store

data_store.check_run_dir(....
```
