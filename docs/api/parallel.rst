Parallel solver (BOC)
=====================

The parallel step partitions the world into independent patches and schedules
the solver across BOC worker sub-interpreters. These modules cover the
scheduler, the spatial partition, and the data marshalled between workers.

bocphysics.parallel
-------------------

.. automodule:: bocphysics.parallel
   :members:

bocphysics.patches
------------------

.. automodule:: bocphysics.patches
   :members:

bocphysics.kernel
-----------------

.. automodule:: bocphysics.kernel
   :members:

bocphysics.transport
--------------------

.. automodule:: bocphysics.transport
   :members:
