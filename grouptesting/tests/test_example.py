# -*- coding: utf-8 -*-

"""UNIT TESTS FOR EXAMPLE

This module contains unit tests for the example module.

"""

from unittest import TestCase
import numpy.testing as npt
from ..example import *
from .. import model, algorithms


class ExampleTestCase(TestCase):

    def setUp(self):

        self.x = 1
        self.y = 2

    def tearDown(self):

        self.x = None
        self.y = None

    def test_add_int(self):

        npt.assert_equal(math.add_int(self.x, self.y), 3,
                         err_msg='Incorrect addition result.')

        npt.assert_raises(TypeError, math.add_int, self.x,
                          float(self.y))


class AlgorithmsTest(TestCase):

    def setUp(self):

        N = 100
        T = 30
        L = 10
        I = 10
        X = model.CCW(N, T, L)
        self.sigma = model.D(N, I)
        y = model.Y(X, sigma)
        self.sigma_hat = algorithms.DD(X, y)
        return

    def tearDown(self):
        return

    def test_add_int(self):

        ## Assert that DD recognizes the definitely defectives.
        npt.assert_equal(self.sigma * self.sigma_hat, self.sigma, err_msg='Incorrect DD result.')
