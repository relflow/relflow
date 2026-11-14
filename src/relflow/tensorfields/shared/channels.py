"""Channel width shared by requests that stack a tuple query."""

from collections.abc import Mapping


def width(data: object) -> object:
    """Fill ``n_dim`` from a tuple query before a request validates.

    Args:
        data: Raw request input, as received by a ``mode="before"`` validator.

    Returns:
        A new mapping with ``n_dim`` set to the number of sources when ``data``
        holds a list or tuple ``query``; any other input unchanged.

    Raises:
        ValueError: If the query names no source, or an explicit ``n_dim`` differs from the number of sources.
    """
    if not isinstance(data, Mapping) or not isinstance(data.get("query"), (list, tuple)):
        return data
    query = tuple(data["query"])
    if not query:
        raise ValueError("tuple query names no source; name at least one source or use a single query")
    n_dim = data.get("n_dim", len(query))
    if n_dim != len(query):
        raise ValueError(
            f"n_dim={n_dim} differs from the {len(query)} sources in query {query!r}; remove n_dim or make it match"
        )
    return {**data, "n_dim": len(query)}


__all__ = ["width"]
