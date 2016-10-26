#!/usr/bin/env python3
import os
import shutil
import sys


def py_ext(filename):
    assert len(filename) > 3
    return filename[:-2] + 'py'


def is_hidden(filename):
    dir, sep, fname = filename.rpartition('/')
    return fname.startswith('.')


def hidden(filename):
    dir, sep, fname = filename.rpartition('/')
    return dir + sep + '.' + fname


def unhidden(filename):
    dir, sep, fname = filename.rpartition('/')
    assert fname.startswith(
        '.'), "Expected file {} to be hidden, but it wasn't.".format(filename)
    return dir + sep + fname[1:]

directory = os.path.dirname(sys.argv[0])
if directory != '.':
    print('Warning! You should only run this script from the \
            top-level directory where it is located, since it \
            will miss some files otherwise, leaving you with a mix of .so and .py files.')
    sure = input('Are you sure you want to run it from here? Y/n: ')
    if sure.lower() != 'y':
        quit()


so_files = []
PY, CY = 0, 1

for dirpath, dirs, filenames in os.walk('.'):
    for f in filenames:
        if f.endswith('.so'):
            so_files.append(dirpath + os.sep + f)

if not so_files:
    print('No Cython-built ".so" files found.')
    quit()

# if so files were hidden before, we will toggle them on
MODE = CY if all([s.rpartition('/')[2].startswith('.')
                  for s in so_files]) else PY
so_files = [unhidden(s) if is_hidden(s) else s for s in so_files]


to_hide = []
to_unhide = []
unchanged = []
unknown = []
if MODE == CY:
    check = input('Switch from Python to Cython? Y/n: ')
    if check.lower() != 'y':
        print('Quitting.')
        quit()
    # hide py, unhide so
    for f in so_files:
        py_ver = py_ext(f)
        if os.path.isfile(py_ver):
            to_hide.append(py_ver)
        elif os.path.isfile(hidden(py_ver)):
            unchanged.append(hidden(py_ver))
        else:
            unknown.append(py_ver)
    for s in so_files:
        if os.path.isfile(hidden(s)):
            to_unhide.append(hidden(s))
        else:
            unchanged.append(s)
elif MODE == PY:
    check = input('Switch from Cython to Python? Y/n: ')
    if check.lower() != 'y':
        print('Quitting.')
        quit()
    # unhide py, hide so
    for s in so_files:
        if os.path.isfile(s):
            to_hide.append(s)
        elif os.path.isfile(hidden(s)):
            unchanged.append(s)
        else:
            unknown.append(s)
    for f in so_files:
        py_ver = hidden(py_ext(f))
        if os.path.isfile(py_ver):
            to_unhide.append(py_ver)
        elif os.path.isfile(unhidden(py_ver)):
            unchanged.append(unhidden(py_ver))
        else:
            unknown.append(py_ver)

if to_hide:
    print('Will hide:')
    for f in to_hide:
        print('\t{}'.format(f))
if to_unhide:
    print('Will unhide:')
    for f in to_unhide:
        print('\t{}'.format(f))
if unknown:
    print('Unaccounted for files, expected but not found:')
    for f in unknown:
        print('\t{}'.format(f))


for f in to_hide:
    shutil.move(f, hidden(f))
for f in to_unhide:
    shutil.move(f, unhidden(f))

print('Done.')
