
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from __future__ import absolute_import
import unittest
import os
import tempfile

from logging import getLogger
import chainsaw.api as api
import numpy as np
from chainsaw.data.numpy_filereader import NumPyFileReader
from chainsaw.data.py_csv_reader import PyCSVReader as CSVReader
import shutil


logger = getLogger('chainsaw.'+'TestReaderUtils')


class TestApiSourceFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        data_np = np.random.random((100, 3))
        data_raw = np.arange(300 * 4).reshape(300, 4)

        cls.dir = tempfile.mkdtemp("test-api-src")

        cls.npy = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.npz = tempfile.mktemp(suffix='.npz', dir=cls.dir)
        cls.dat = tempfile.mktemp(suffix='.dat', dir=cls.dir)
        cls.csv = tempfile.mktemp(suffix='.csv', dir=cls.dir)

        cls.bs = tempfile.mktemp(suffix=".txt", dir=cls.dir)

        with open(cls.bs, "w") as fh:
            fh.write("meaningless\n")
            fh.write("this can not be interpreted\n")

        np.save(cls.npy, data_np)
        np.savez(cls.npz, data_np, data_np)
        np.savetxt(cls.dat, data_raw)
        np.savetxt(cls.csv, data_raw)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_obtain_numpy_file_reader_npy(self):
        reader = api.source(self.npy)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(
            isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")

    @unittest.skip("npz currently unsupported")
    def test_obtain_numpy_file_reader_npz(self):
        reader = api.source(self.npz)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(
            isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")

    def test_obtain_csv_file_reader_dat(self):
        reader = api.source(self.dat)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")

    def test_obtain_csv_file_reader_csv(self):
        reader = api.source(self.csv)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")

    def test_bullshit_csv(self):
        # this file is not parseable as tabulated float file
        with self.assertRaises(ValueError) as exc:
            api.source(self.bs)

        self.assertIn("could not parse", exc.exception.args[0])


if __name__ == "__main__":
    unittest.main()