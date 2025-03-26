import tilelang

# Dictionary to store cached kernels
_cached = {}


def cached(
        func,
        out_idx = None,
        *args,
        target = "auto",
        target_host = None,
):
    """
    Cache and reuse compiled kernels to avoid redundant compilation.

    Args:
        func: Function to be compiled or a PrimFunc that's already prepared
        out_idx: Indices specifying which outputs to return
        target: Compilation target platform
        target_host: Host target for compilation
        *args: Arguments passed to func when calling it

    Returns:
        JITKernel: The compiled kernel, either freshly compiled or from cache
    """
    global _cached
    # Create a unique key based on the function, output indices and arguments
    key = (func, tuple(out_idx), *args)

    # Return cached kernel if available
    if key not in _cached:
        # Handle both PrimFunc objects and callable functions
        # program = func if isinstance(func, PrimFunc) else func(*args)
        program = func(*args)

        # Compile the program to a kernel
        kernel = tilelang.compile(program, out_idx=out_idx, target=target, target_host=target_host)
        # Store in cache for future use
        _cached[key] = kernel

    return _cached[key]
