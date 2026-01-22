Performance and System Stability
================================

Large-scale analyses in mdxplain can stress a machine in ways that are unusual
for everyday scientific software. This page explains **what we observed**,
**why this can happen**, and **what mdxplain does to reduce the risk of system
freezes**.

The goal is not to scare users, but to make heavy jobs **predictably slow
instead of unpredictably unstable**.

This explanation is written for computational scientists and bioinformaticians,
not for operating-system specialists. Technical terms are explained when they
first appear.

--------------------------------------------------------------------------

What we observed
----------------

When running very large jobs (many samples, very large intermediate matrices or
graphs, large memory-mapped files, and multiple full passes over the data), we
repeatedly observed the following behavior:

- The system can become **completely unresponsive** during intensive read/write
  phases. The graphical interface stops updating, terminals do not respond, and
  SSH connections may freeze.

- Disk activity appears **fully saturated**: other programs (desktop, logging,
  remote login) compete for the same storage and become extremely slow or stop
  responding.

- CPU cores are busy, but the machine does not feel "just slow" — it feels
  **frozen**, as if input events are no longer processed.

- In rare but reproducible cases (for example by running kernel PCA or diffusion
  maps on very large simulations), the system was so blocked that **even long
  pressing the power button did not turn the machine off**. Recovery was only
  possible by cutting power completely.

These effects were not random. They appeared consistently at large simulations
(over 200k frames and 5k+ atoms) during heavy computations such as kernel PCA or
diffusion maps, indicating real **stability limits** rather than individual
hardware failures.

--------------------------------------------------------------------------

Why this happens
----------------

To understand these freezes, it helps to know how large numerical workloads
interact with the operating system.

Memory-mapped files and the page cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mdxplain uses *memory-mapped files* (often called *memmaps*) for large
intermediate arrays. A memmap lets the program treat data on disk as if it were
in memory, while the operating system transparently loads and writes parts of
the file as needed.

The operating system keeps recently used data in a **page cache** (RAM used as a
buffer for disk I/O). When the program writes to a memmap, modified pages become
*dirty pages* — they must eventually be written back to disk.

If a program writes large amounts of data very quickly, many dirty pages can
accumulate. When internal limits are reached, the operating system forces a
large **writeback** operation (flushing data to disk). During this phase, many
processes can be blocked, and the system may feel frozen.

--------------------------------------------------------------------------

The pressure of approximations (e.g., Nyström)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even when using memory-efficient methods such as the **Nyström approximation**,
system stability is not guaranteed. While Nyström avoids the O(N²) memory
explosion, the construction phase introduces distinct sources of pressure:

1. **Landmark selection:** Selecting landmarks requires full linear scans of the
   original trajectory. For very large datasets, these scans can evict large
   parts of the page cache, forcing system services and other applications out
   of memory.

2. **Address translation overhead:** Accessing large matrices (even N × m) in
   partially irregular patterns increases pressure on the CPU’s memory
   translation machinery (for example the Translation Lookaside Buffer, TLB).
   A high rate of page faults shifts CPU time from numerical work to memory
   management in the kernel.

--------------------------------------------------------------------------

Disk I/O saturation
~~~~~~~~~~~~~~~~~~~

Large analyses often read and write hundreds of gigabytes. If the storage device
(SSD or HDD) is fully busy, all programs that need disk access are delayed.
This includes system services, logging, desktop components, and remote login.

When disk queues are full, the system can appear unresponsive even if CPU and
memory are still available.

--------------------------------------------------------------------------

Thrashing
~~~~~~~~~

In extreme cases, the combination of large working sets, repeated full passes
over the data, and concurrent parallel workers can push the system into a state
known as *thrashing*. In this state, the operating system spends most of its
time moving memory pages between RAM and disk instead of making forward
progress, which can make the entire machine appear frozen.

--------------------------------------------------------------------------

Too much parallelism (oversubscription)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many numerical libraries (NumPy, SciPy, BLAS, LAPACK) use multiple CPU threads
internally. At the same time, mdxplain or libraries it depends on (such as
scikit-learn) may run multiple processes in parallel.

If both levels are active, the system can end up with **far more active threads
than CPU cores**. This is called *oversubscription*.

Oversubscription causes heavy context switching, cache thrashing, and additional
I/O pressure, making freezes more likely.

Some libraries, such as scikit-learn, already use ``threadpoolctl`` internally to
avoid oversubscription within their own code. However, oversubscription can
still occur when multiple layers of parallelism interact (for example,
process-based parallelism combined with threaded numerical libraries, or a mix
of different libraries with independent thread pools). mdxplain therefore
applies additional global limits to ensure that the total number of active
threads remains bounded.

--------------------------------------------------------------------------

Access pattern confusion
~~~~~~~~~~~~~~~~~~~~~~~~

Some phases of an analysis read data sequentially (linear scans), while others
access data in a more random pattern (graphs, neighborhoods, iterative solvers).
The operating system attempts to infer the access pattern automatically, but it
can make suboptimal decisions.

If this happens, unnecessary disk reads and cache evictions can significantly
slow the system.

--------------------------------------------------------------------------

Why even the power button may not work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On many modern systems, the power button is not a purely mechanical hardware
switch. Instead, it generates an event that is handled through firmware and the
operating system, often in user space (for example by ``systemd-logind`` on
Linux).

Under extreme conditions—such as sustained disk I/O saturation, memory reclaim,
or thrashing—handling of such events can be delayed. This can make the machine
appear completely unresponsive.

In very rare situations, prolonged resource starvation may contribute to
*livelock-like* behavior, where progress is prevented despite ongoing activity.
In such cases, even firmware-mediated mechanisms involved in power-button
handling may not react immediately, requiring a full power cut to recover.

--------------------------------------------------------------------------

How mdxplain reduces the risk
-----------------------------

mdxplain cannot make extremely large jobs cheap, but it can make them **much more
predictable and safer to run**. The following measures are applied automatically
or when stability mode is enabled.

Chunk-level flushing of memmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large memmap writes are flushed to disk **in small chunks** (for example via
``msync``) instead of all at once. This reduces the buildup of dirty pages and
prevents sudden global writeback stalls.

--------------------------------------------------------------------------

Explicit access-pattern hints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before large linear scans, mdxplain informs the operating system (via
``madvise``) that access will be sequential. Afterward, this hint is reset for
random-access phases. This improves caching behavior during large streaming
passes.

--------------------------------------------------------------------------

Controlled parallelism
~~~~~~~~~~~~~~~~~~~~~~

mdxplain limits the number of threads used by numerical libraries to avoid
oversubscription. When process-based parallelism is active, internal threading
may be reduced to one thread per process.

--------------------------------------------------------------------------

CPU and I/O fairness
~~~~~~~~~~~~~~~~~~~~

When supported by the platform, mdxplain lowers its CPU and disk scheduling
priority (for example using ``nice`` and ``ionice``). This helps ensure that the
operating system, desktop, and interactive processes remain responsive even
under heavy load.

--------------------------------------------------------------------------

Platform safety and HPC environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All resource controls are **best-effort**. If a platform does not support a
specific control, mdxplain logs a warning and continues safely.

On HPC systems, mdxplain respects scheduler limits (CPU sets, cgroups) and never
uses resources outside the job allocation.

--------------------------------------------------------------------------

Summary
-------

Large-scale analyses can push a system into extreme resource-pressure states.
Without care, this can lead to complete system freezes. mdxplain actively manages
memory, I/O, and parallelism to reduce these risks, prioritizing system stability
over raw throughput for large jobs.

--------------------------------------------------------------------------

References
----------

- Linux virtual memory and writeback behavior:
  https://docs.kernel.org/admin-guide/sysctl/vm.html

- Memory pressure, page reclaim, and thrashing (background):
  https://gist.github.com/JPvRiel/bcc5b20aac0c9cce6eefa6b88c125e03

- ``madvise`` system call and access pattern hints:
  https://man7.org/linux/man-pages/man2/madvise.2.html

- Thread pool control and oversubscription:
  https://github.com/joblib/threadpoolctl

- Parallelism in scientific Python:
  https://scikit-learn.org/stable/computing/parallelism.html

- systemd power button handling:
  https://www.freedesktop.org/software/systemd/man/logind.conf.html

- Livelock and resource starvation:
  https://en.wikipedia.org/wiki/Livelock
