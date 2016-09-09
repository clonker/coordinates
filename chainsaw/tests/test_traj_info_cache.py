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
'''
Created on 30.04.2015

@author: marscher
'''

from __future__ import absolute_import

from tempfile import NamedTemporaryFile

import os
import tempfile
import unittest

import mock
import chainsaw
from chainsaw import api
from chainsaw.data.numpy_filereader import NumPyFileReader
from chainsaw.data.py_csv_reader import PyCSVReader
from chainsaw.data.util.traj_info_backends import SqliteDB
from chainsaw.data.util.traj_info_cache import TrajectoryInfoCache

from chainsaw import config

from chainsaw.util.contexts import settings
from chainsaw.util.files import TemporaryDirectory
import pkg_resources

import numpy as np

try:
    import pyemma
    have_feature_reader = True
except ImportError:
    have_feature_reader = False


class TestTrajectoryInfoCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_instance = TrajectoryInfoCache.instance()
        config.use_trajectory_lengths_cache = True

        if have_feature_reader:
            from pyemma.datasets import get_bpti_test_data
            d = get_bpti_test_data()
            cls.xtcfiles =d['trajs']
            cls.pdbfile = d['top']

    def setUp(self):
        self.work_dir = tempfile.mkdtemp("traj_cache_test")
        self.tmpfile = tempfile.mktemp(dir=self.work_dir)
        self.db = TrajectoryInfoCache(self.tmpfile)

        # overwrite TrajectoryInfoCache._instance with self.db...
        TrajectoryInfoCache._instance = self.db

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmpfile)

        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        TrajectoryInfoCache._instance = cls.old_instance
        config.use_trajectory_lengths_cache = False

    def test_get_instance(self):
        # test for exceptions in singleton creation
        inst = TrajectoryInfoCache.instance()
        inst.current_db_version
        self.assertIs(inst, self.db)

    def test_store_load_traj_info(self):
        x = np.random.random((10, 3))
        my_conf = config()
        my_conf.cfg_dir = self.work_dir
        with mock.patch('chainsaw.data.util.traj_info_cache.config', my_conf):
            with NamedTemporaryFile(delete=False) as fh:
                np.savetxt(fh.name, x)
                reader = api.source(fh.name)
                info = self.db[fh.name, reader]
                self.db.close()
                self.db.__init__(self.db._database.filename)
                info2 = self.db[fh.name, reader]
                self.assertEqual(info2, info)

    def test_exceptions(self):
        # in accessible files
        not_existant = ''.join(
            chr(i) for i in np.random.randint(65, 90, size=10)) + '.npy'
        bad = [not_existant]  # should be unaccessible or non existent
        with self.assertRaises(ValueError) as cm:
            api.source(bad)
            assert bad[0] in cm.exception.message

        # empty files
        with NamedTemporaryFile(delete=False) as f:
            f.close()
            with self.assertRaises(ValueError) as cm:
                api.source(f.name)
                assert f.name in cm.exception.message

    @unittest.skipIf(not have_feature_reader, "dont have feature reader")
    def test_featurereader_xtc(self):
        import mdtraj
        # cause cache failures
        from pyemma.coordinates.data import FeatureReader
        with settings(use_trajectory_lengths_cache=False):
            reader = FeatureReader(self.xtcfiles, self.pdbfile)

        results = {}
        for f in self.xtcfiles:
            traj_info = self.db[f, reader]
            results[f] = traj_info.ndim, traj_info.length, traj_info.offsets

        expected = {}
        for f in self.xtcfiles:
            with mdtraj.open(f) as fh:
                length = len(fh)
                ndim = fh.read(1)[0].shape[1]
                offsets = fh.offsets if hasattr(fh, 'offsets') else []
                expected[f] = ndim, length, offsets

        np.testing.assert_equal(results, expected)

    def test_npy_reader(self):
        lengths_and_dims = [(7, 3), (23, 3), (27, 3)]
        data = [
            np.empty((n, dim)) for n, dim in lengths_and_dims]
        files = []
        with TemporaryDirectory() as td:
            for i, x in enumerate(data):
                fn = os.path.join(td, "%i.npy" % i)
                np.save(fn, x)
                files.append(fn)

            reader = NumPyFileReader(files)

            # cache it and compare
            results = {f: (self.db[f, reader].length, self.db[f, reader].ndim,
                           self.db[f, reader].offsets) for f in files}
            expected = {f: (len(data[i]), data[i].shape[1], [])
                        for i, f in enumerate(files)}
            np.testing.assert_equal(results, expected)

    def test_csvreader(self):
        data = np.random.random((101, 3))
        fn = tempfile.mktemp()
        try:
            np.savetxt(fn, data)
            # calc offsets
            offsets = [0]
            with open(fn, PyCSVReader.DEFAULT_OPEN_MODE) as new_fh:
                while new_fh.readline():
                    offsets.append(new_fh.tell())
            reader = PyCSVReader(fn)
            assert reader.dimension() == 3
            trajinfo = reader._get_traj_info(fn)
            np.testing.assert_equal(offsets, trajinfo.offsets)
        finally:
            os.unlink(fn)

    @unittest.skipIf(not have_feature_reader, "dont have feature reader")
    def test_fragmented_reader(self):
        top_file = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        trajfiles = []
        nframes = []
        with TemporaryDirectory() as wd:
            for _ in range(3):
                from pyemma.coordinates.tests.util import create_traj
                f, _, l = create_traj(top_file, dir=wd)
                trajfiles.append(f)
                nframes.append(l)
            # three trajectories: one consisting of all three, one consisting of the first,
            # one consisting of the first and the last
            reader = api.source(
                [trajfiles, [trajfiles[0]], [trajfiles[0], trajfiles[2]]], top=top_file)
            np.testing.assert_equal(reader.trajectory_lengths(),
                                    [sum(nframes), nframes[0], nframes[0] + nframes[2]])

    @unittest.skipIf(not have_feature_reader, "dont have feature reader")
    def test_feature_reader_xyz(self):
        import mdtraj
        traj = mdtraj.load(self.xtcfiles, top=self.pdbfile)
        length = len(traj)

        with NamedTemporaryFile(mode='wb', suffix='.xyz', delete=False) as f:
            fn = f.name
            traj.save_xyz(fn)
            f.close()
            reader = chainsaw.source(fn, top=self.pdbfile)
            self.assertEqual(reader.trajectory_length(0), length)

    def test_data_in_mem(self):
        # make sure cache is not used for data in memory!
        data = [np.empty((3, 3))] * 3
        api.source(data)
        self.assertEqual(self.db.num_entries, 0)

    def test_old_db_conversion(self):
        # prior 2.1, database only contained lengths (int as string) entries
        # check conversion is happening
        with NamedTemporaryFile(suffix='.npy', delete=False) as f:
            db = TrajectoryInfoCache(None)
            fn = f.name
            np.save(fn, [1, 2, 3])
            f.close()  # windows sucks
            reader = api.source(fn)
            hash = db._get_file_hash(fn)
            from chainsaw.data.util.traj_info_backends import DictDB
            db._database = DictDB()
            db._database.db_version = 0

            info = db[fn, reader]
            assert info.length == 3
            assert info.ndim == 1
            assert info.offsets == []

    @unittest.skipIf(not have_feature_reader, "dont have feature reader")
    def test_corrupted_db(self):
        with NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("makes no sense!!!!")
            f.close()
        name = f.name
        import warnings
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter('always')
            db = TrajectoryInfoCache(name)
            assert len(cm) == 1
            assert "corrupted" in str(cm[-1].message)

        # ensure we can perform lookups on the broken db without exception.
        r = api.source(self.xtcfiles[0], top=self.pdbfile)
        db[self.xtcfiles[0], r]

    def test_n_entries(self):
        self.assertEqual(self.db.num_entries, 0)
        assert TrajectoryInfoCache._instance is self.db
        chainsaw.source(self.xtcfiles, top=self.pdbfile)
        self.assertEqual(self.db.num_entries, len(self.xtcfiles))

    def test_max_n_entries(self):
        data = [np.random.random((10, 3)) for _ in range(20)]
        max_entries = 10
        config.traj_info_max_entries = max_entries
        files = []
        with TemporaryDirectory() as td:
            for i, arr in enumerate(data):
                f = os.path.join(td, "%s.npy" % i)
                np.save(f, arr)
                files.append(f)
            chainsaw.source(files)
        self.assertLessEqual(self.db.num_entries, max_entries)
        self.assertGreater(self.db.num_entries, 0)

    def test_max_size(self):
        data = [np.random.random((150, 10)) for _ in range(150)]
        max_size = 1

        files = []
        config.show_progress_bars = False
        with TemporaryDirectory() as td, settings(traj_info_max_size=max_size):
            for i, arr in enumerate(data):
                f = os.path.join(td, "%s.txt" % i)
                # save as txt to enforce creation of offsets
                np.savetxt(f, arr)
                files.append(f)
            chainsaw.source(files)

        self.assertLessEqual(os.stat(self.db.database_filename).st_size / 1024, config.traj_info_max_size)
        self.assertGreater(self.db.num_entries, 0)

    def test_no_working_directory(self):
        # this is the case as long as the user has not yet created a config directory via config.save()
        self.db._database = SqliteDB(filename=None)

        # trigger caching
        chainsaw.source(self.xtcfiles, top=self.pdbfile)

    def test_no_sqlite(self):
        # create new instance (init has to be called, install temporary import hook to raise importerror for sqlite3
        import sys
        del sys.modules['sqlite3']

        class meta_ldr(object):
            def find_module(self, fullname, path):
                if fullname.startswith('sqlite3'):
                    return self

            def load_module(self, fullname, path=None):
                raise ImportError()

        import warnings
        try:
            sys.meta_path.insert(0, meta_ldr())
            # import sqlite3
            with warnings.catch_warnings(record=True) as cw:
                db = TrajectoryInfoCache()
                self.assertNotIsInstance(db._database, SqliteDB)
            self.assertEqual(len(cw), 1)
            self.assertIn("sqlite3 package not available", cw[0].message.args[0])
        finally:
            del sys.meta_path[0]

    @unittest.skipIf(not have_feature_reader, "dont have feature reader")
    def test_in_memory_db(self):
        """ new instance, not yet saved to disk, no lru cache avail """
        old_cfg_dir = config.cfg_dir
        try:
            config._cfg_dir = ''
            db = TrajectoryInfoCache()
            reader = chainsaw.source(self.xtcfiles, top=self.pdbfile)

            info = db[self.xtcfiles[0], reader]
            self.assertIsInstance(db._database, SqliteDB)

            directory = db._database._database_from_key(info.hash_value)
            assert directory is None
        finally:
            from chainsaw.util.exceptions import ConfigDirectoryException
            try:
                config.cfg_dir = old_cfg_dir
            except ConfigDirectoryException:
                pass

if __name__ == "__main__":
    unittest.main()
