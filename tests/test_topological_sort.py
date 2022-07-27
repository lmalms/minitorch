from minitorch.autodiff import Scalar, topological_sort


def test_topological_sort1():
    scalar1, scalar2 = 1.5, 2.5
    scalar3 = scalar1 + scalar2

    sorted_ = topological_sort(scalar3)
    assert len(sorted_) == 0


def test_topological_sort2():
    scalar1 = 1.5
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 + scalar2

    sorted_ = topological_sort(scalar3)
    assert len(sorted_) == 1
    assert sorted_[0].name == scalar2.name


def test_topological_sort3():
    scalar1 = 1.5
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 + scalar2
    scalar4 = Scalar(4.5)
    scalar5 = scalar3 + scalar4

    sorted_ = topological_sort(scalar5)
    assert len(sorted_) == 3
    assert sorted_[0].name == scalar3.name
    assert sorted_[1].name == scalar2.name
    assert sorted_[2].name == scalar4.name


def test_topological_sort4():
    scalar1 = Scalar(1.5)
    scalar2 = Scalar(2.5)
    scalar3 = scalar1 * scalar2
    scalar4 = scalar3.log()
    scalar5 = scalar3.exp()
    scalar6 = scalar4 + scalar5

    sorted_ = topological_sort(scalar6)
    assert len(sorted_) == 5
    assert sorted_[0].name == scalar4.name
    assert sorted_[1].name == scalar3.name
    assert sorted_[2].name == scalar1.name
