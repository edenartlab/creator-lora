def chunk_list(list_to_be_chunked, chunk_size):
    """
    splits a list into chunks of size n
    """
    chunk_size = max(1, chunk_size)
    return list(
        (
            list_to_be_chunked[i : i + chunk_size]
            for i in range(0, len(list_to_be_chunked), chunk_size)
        )
    )