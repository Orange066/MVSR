On version numbers
^^^^^^^^^^^^^^^^^^

The two version numbers (C++ and Python) must match when combined (checked when
you build the PyPI package), and must be a valid `PEP 440
<https://www.python.org/dev/peps/pep-0440>`_ version when combined.

For example:

.. code-block:: C++

    #define PYBIND11_VERSION_MAJOR X
    #define PYBIND11_VERSION_MINOR Y
    #define PYBIND11_VERSION_PATCH Z.dev1

For beta, ``PYBIND11_VERSION_PATCH`` should be ``Z.b1``. RC's can be ``Z.rc1``.
Always include the dot (even though PEP 440 allows it to be dropped). For a
final release, this must be a simple integer.


To release a new version of pybind11:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Update the version number
  - Update ``PYBIND11_VERSION_MAJOR`` etc. in
    ``include/pybind11/detail/common.h``. PATCH should be a simple integer.
  - Update ``pybind11/_version.py`` (match above)
  - Ensure that all the information in ``setup.py`` is up-to-date.
  - Add release date in ``docs/changelog.rst``.
  - ``git add`` and ``git commit``, ``git push``. **Ensure CI passes**. (If it
    fails due to a known flake issue, either ignore or restart CI.)
- Add a release branch if this is a new minor version
  - ``git checkout -b vX.Y``, ``git push -u origin vX.Y``
- Update tags
  - ``git tag -a vX.Y.Z -m 'vX.Y.Z release'``.
  - ``git push --tags``.
- Update stable
    - ``git checkout stable``
    - ``git merge master``
    - ``git push``
- Make a GitHub release (this shows up in the UI, sends new release
  notifications to users watching releases, and also uploads PyPI packages).
  (Note: if you do not use an existing tag, this creates a new lightweight tag
  for you, so you could skip the above step).
  - GUI method: click "Create a new release" on the far right, fill in the tag
    name, fill in a release name like "Version X.Y.Z", and optionally
    copy-and-paste the changelog into the description (processed as markdown by
    Pandoc). Check "pre-release" if this is a beta/RC.
  - CLI method: with ``gh`` installed, run ``gh release create vX.Y.Z -t "Version X.Y.Z"``
    If this is a pre-release, add ``-p``.

- Get back to work
  - Make sure you are on master, not somewhere else: ``git checkout master``
  - Update version macros in ``include/pybind11/common.h`` (set PATCH to
    ``0.dev1`` and increment MINOR).
  - Update ``_version.py`` to match
  - Add a plot for in-development updates in ``docs/changelog.rst``.
  - ``git add``, ``git commit``, ``git push``

If a version branch is updated, remember to set PATCH to ``1.dev1``.


Manual packaging
^^^^^^^^^^^^^^^^

If you need to manually upload releases, you can download the releases from the job artifacts and upload them with twine. You can also make the files locally (not recommended in general, as your local directory is more likely to be "dirty" and SDists love picking up random unrelated/hidden files); this is the procedure:

.. code-block:: bash

    python3 -m pip install build
    python3 -m build
    PYBIND11_SDIST_GLOBAL=1 python3 -m build
    twine upload dist/*

This makes SDists and wheels, and the final line uploads them.
