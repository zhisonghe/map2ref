import sys
import types


def _install_pandas_legacy_index_shim():
    """Allow unpickling pretrained models saved under pandas<2.0.

    pandas 2.0 removed the Int64Index/UInt64Index/Float64Index/NumericIndex
    classes (merged into Index), so torch.load() of older scvi-tools/scArches
    checkpoints fails with ModuleNotFoundError: pandas.core.indexes.numeric.
    """
    module_name = 'pandas.core.indexes.numeric'
    if module_name in sys.modules:
        return
    import pandas as pd
    shim = types.ModuleType(module_name)
    for name in ('Int64Index', 'UInt64Index', 'Float64Index', 'NumericIndex'):
        setattr(shim, name, pd.Index)
    sys.modules[module_name] = shim


_install_pandas_legacy_index_shim()
