
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

import os

from six import string_types


def create_file_reader(input_files, topology, featurizer, chunk_size=1000, **kw):
    r"""
    Creates a (possibly featured) file reader by a number of input files and either a topology file or a featurizer.
    Parameters
    ----------
    :param input_files:
        A single input file or a list of input files.
    :param topology:
        A topology file. If given, the featurizer argument can be None.
    :param featurizer:
        A featurizer. If given, the topology file can be None.
    :param chunk_size:
        The chunk size with which the corresponding reader gets initialized.
    :return: Returns the reader.
    """
    # fragmented trajectories
    if (isinstance(input_files, (list, tuple)) and len(input_files) > 0 and
            any(isinstance(item, (list, tuple)) for item in input_files)):
        from chainsaw.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
        return FragmentedTrajectoryReader(input_files, topology, chunk_size, featurizer)

    # normal trajectories
    if (isinstance(input_files, string_types)
            or (isinstance(input_files, (list, tuple))
                and (any(isinstance(item, string_types) for item in input_files) 
                     or len(input_files) is 0))):
        reader = None
        # check: if single string create a one-element list
        if isinstance(input_files, string_types):
            input_list = [input_files]
        elif len(input_files) > 0 and all(isinstance(item, string_types) for item in input_files):
            input_list = input_files
        else:
            if len(input_files) is 0:
                raise ValueError("The passed input list should not be empty.")
            else:
                raise ValueError("The passed list did not exclusively contain strings or was a list of lists "
                                 "(fragmented trajectory).")

        _, suffix = os.path.splitext(input_list[0])

        # check: do all files have the same file type? If not: raise ValueError.
        if all(item.endswith(suffix) for item in input_list):

            # do all the files exist? If not: Raise value error
            all_exist = True
            err_msg = ""
            for item in input_list:
                if not os.path.isfile(item):
                    err_msg += "\n" if len(err_msg) > 0 else ""
                    err_msg += "File %s did not exist or was no file" % item
                    all_exist = False
            if not all_exist:
                raise ValueError("Some of the given input files were directories"
                                 " or did not exist:\n%s" % err_msg)

            if all_exist:
                from chainsaw.data.util.fileformat_registry import FileFormatRegistry
                # check all registered suffixes.
                if suffix in FileFormatRegistry.supported_extensions():
                    clazz = FileFormatRegistry[suffix]
                    if FileFormatRegistry.is_md_format(suffix):
                        # check: do we either have a featurizer or a topology file name? If not: raise ValueError.
                        # create a MD reader with file names and topology
                        if not featurizer and not topology:
                            raise ValueError("The input files were MD files which makes it mandatory to have either a "
                                             "featurizer or a topology file.")
                        reader = clazz(input_list, featurizer=featurizer, topologyfile=topology, chunksize=chunk_size)
                    else:
                        reader = clazz(input_list, chunksize=chunk_size, **kw)
                else:
                    import pprint
                    exts = pprint.pformat(sorted(FileFormatRegistry.supported_extensions()))
                    raise ValueError("Extension {ext} not supported. Supported extensions are:\n{exts}"
                                     .format(ext=suffix, exts=exts))
        else:
            raise ValueError("Not all elements in the input list were of the type %s!" % suffix)
    else:
        raise ValueError("Input \"%s\" was no string or list of strings." % input)
    return reader
