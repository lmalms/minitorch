from minitorch.autodiff import Scalar, topological_sort

import pytest


@pytest.mark.skip(reason="topological search uses bfs instead of dfs")
def test_dfs_topological_sort1():
    scalar1, scalar2 = 1.5, 2.5
    scalar3 = scalar1 + scalar2

    sorted_ = topological_sort(scalar3)
    assert len(sorted_) == 0


@pytest.mark.skip(reason="topological search uses bfs instead of dfs")
def test_dfs_topological_sort2():
    scalar1 = 1.5
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 + scalar2

    sorted_ = topological_sort(scalar3)
    assert len(sorted_) == 2
    assert sorted_[0].name == scalar3.name
    assert sorted_[1].name == scalar2.name


@pytest.mark.skip(reason="topological search uses bfs instead of dfs")
def test_dfs_topological_sort3():
    scalar1 = 1.5
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 + scalar2
    scalar4 = Scalar(4.5)
    scalar5 = scalar3 + scalar4

    sorted_ = topological_sort(scalar5)
    assert len(sorted_) == 4
    assert sorted_[0].name == scalar5.name
    assert sorted_[1].name == scalar3.name
    assert sorted_[2].name == scalar2.name
    assert sorted_[3].name == scalar4.name


@pytest.mark.skip(reason="topological search uses bfs instead of dfs")
def test_dfs_topological_sort4():
    scalar1 = Scalar(1.5)
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 * scalar2
    scalar4 = scalar3.log()
    scalar5 = scalar3.exp()
    scalar6 = scalar4 + scalar5

    sorted_ = topological_sort(scalar6)
    assert len(sorted_) == 6
    assert sorted_[0].name == scalar6.name
    assert sorted_[1].name == scalar4.name
    assert sorted_[2].name == scalar3.name
    assert sorted_[-1].name == scalar5.name


