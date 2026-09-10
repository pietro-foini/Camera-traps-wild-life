# Contributing to Camera-traps-wild-life

Welcome! 😀 

This guide will help you contribute code, docs, and fixes to the project smoothly.

## Docstring guidance

We use [reStructuredText (reST) / Sphinx style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for docstrings:

- Use `:param <name>: <description>` and `:type <name>: <type>` for each argument.
- Use `:return:` and `:rtype:` to describe return values.
- Keep descriptions concise and clear.

Example:

```python
def accuracy(tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0) -> float | None:
    """
    Compute accuracy.

    :param tp: number of true positives
    :type tp: int
    :param fp: number of false positives
    :type fp: int
    :param fn: number of false negatives
    :type fn: int
    :param tn: number of true negatives
    :type tn: int
    :return: accuracy score as a float rounded to two decimal places
    :rtype: float | None
    """
    
    ...
```

---

## Type hints guidance

- Avoid unnecessary use of `typing.cast`; prefer refactors that naturally convey types to type-checkers.
- Avoid overly complex nested types (e.g. `list[list[dict[str, str]]]`). Prefer defining clear type aliases or data models for readability.
- Use builtin generics (`list`, `dict`) when possible.

---

## Commits guidance

We follow a simple [Git Commit Convention](https://www.conventionalcommits.org/en/v1.0.0/) to keep history clear:

    <type>: <short description>

Common types:

- **feat**: new feature or enhancement  
- **fix**: bug fix  
- **docs**: documentation changes  
- **chore**: maintenance, refactoring, or non-functional changes  
- **test**: add or update tests

Example:

    docs: add commit guidance section

This project uses Git hooks managed with `pre-commit` to automatically check code quality before each commit.
