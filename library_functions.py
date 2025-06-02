import itertools
import numpy as np
from typing import List, Tuple

# Placeholder for variable names and presets
default_max_degree = 3
var_names = []
function_presets = {}


def generate_registry(n_eq: int, max_degree: int = default_max_degree):
    """
    Generate monomial and trigonometric/exponential/log features and naming functions.
    """
    global var_names, function_presets

    # 1) define variable names
    var_names = [chr(ord('u') + i) for i in range(n_eq)]

    # 2) clear old dynamic definitions
    for name in list(globals()):
        if name.startswith("f_") or name.startswith("s_f_"):
            del globals()[name]

    # 3) monomials up to max_degree
    for degree in range(1, max_degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_eq), degree):
            names = [var_names[i] for i in combo]
            fname = "f_" + "".join(names)
            sname = "s_f_" + "".join(names)
            # feature function
            src_f = f"def {fname}({', '.join(var_names)}):\n    return " + " * ".join(names)
            exec(src_f, globals())
            # naming function
            src_s = f"def {sname}({', '.join(var_names)}):\n    return {repr(''.join(names))}"
            exec(src_s, globals())

    # 4) add sin, cos, exp, ln for each variable
    funcs_map = {
        "sin": "np.sin",
        "cos": "np.cos",
        "exp": "np.exp",
        "ln":  "np.log"
    }
    for var in var_names:
        for suffix, npfunc in funcs_map.items():
            fname = f"f_{suffix}_{var}"
            sname = f"s_f_{suffix}_{var}"
            # feature function
            src_f = f"def {fname}({', '.join(var_names)}):\n    return {npfunc}({var})"
            exec(src_f, globals())
            # naming function
            src_s = f"def {sname}({', '.join(var_names)}):\n    return {repr(f'{suffix}({var})')}"
            exec(src_s, globals())

    # 5) rebuild function_presets with monomials
    degree_labels = {1: "linear", 2: "quadratic", 3: "cubic", 4: "fourth"}
    function_presets = {}
    for deg, label in degree_labels.items():
        combos = itertools.combinations_with_replacement(range(n_eq), deg)
        function_presets[label] = [
            globals()[f"f_{''.join(var_names[i] for i in combo)}"]
            for combo in combos
        ]
    # exotic preset for sin/cos/exp/ln
    exotic = []
    for var in var_names:
        for suffix in funcs_map.keys():
            exotic.append(globals()[f"f_{suffix}_{var}"])
    function_presets["exotic"] = exotic

    # combined preset
    function_presets["all"] = sum(function_presets.values(), [])


def get_functions_and_naming_functions(
    presets: List[str], functions: List[str]
) -> Tuple[List, List]:
    """
    Given a list of preset names and extra function names, return two lists:
      - feature_functions (f_*)
      - naming_functions  (s_f_*) in same order
    """
    funcs = []
    names = []
    for preset in presets:
        for fn in function_presets.get(preset, []):
            funcs.append(fn)
            names.append(globals()[f"s_{fn.__name__}"])
    for fn_name in functions:
        funcs.append(globals()[fn_name])
        names.append(globals()[f"s_{fn_name}"])
    # de-duplicate while preserving order
    funcs = list(dict.fromkeys(funcs))
    names = list(dict.fromkeys(names))
    return funcs, names
