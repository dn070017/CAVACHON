"""Shared mocks for cavachon.layers.modifiers tests.

The only cross-package dependency of the modifiers is
``cavachon.utils.tensor_utils.TensorUtils`` (``is_sparse_tensor``). It is
replaced by a MagicMock at its lookup site in each modifier module, with a
stand-in return value (True only for ``tf.SparseTensor``).
``cavachon.environment.Constants`` is plain constants and stays real.
"""
import importlib
from unittest import mock

import pytest
import tensorflow as tf


def _fake_is_sparse(t):
    return isinstance(t, tf.SparseTensor)


@pytest.fixture
def mock_tensor_utils(monkeypatch):
    """Factory: patch ``TensorUtils`` in ``cavachon.layers.modifiers.<name>``.

    Returns the MagicMock so tests can assert on / override
    ``is_sparse_tensor``.
    """

    def _patch(module_name: str) -> mock.MagicMock:
        module = importlib.import_module(f"cavachon.layers.modifiers.{module_name}")
        fake = mock.MagicMock(name=f"{module_name}.TensorUtils")
        fake.is_sparse_tensor.side_effect = _fake_is_sparse
        monkeypatch.setattr(module, "TensorUtils", fake)
        return fake

    return _patch
