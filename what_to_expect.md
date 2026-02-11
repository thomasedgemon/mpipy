## What to expect with distributed computing


- If you are new to distributed computation, then you should know it is not always a given that a cluster will perform a task faster than a single computer - even if the single computer has exactly the same specs as the worker node computers in the cluster. 
Distribution of a task comes with the unavoidable downside of communication overhead. This can be mitigated by:
1. different communication types - ethernet vs infiniband vs nvlink
2. different communication speeds - 2.5Gbps NICs and switches, rather than 1Gbps 
3. smarter (read: minimal) communication practices

or some combination of these. 

Ethernet/RJ45/SSH/TCP is many orders of magnitude slower than RAM and CPU cache - even if that cable from the switch to the machine is only an inch long. If you have a task that takes ten seconds on a single computer, a cluster is not going to be any faster, and in fact will probably be considerably slower due to aforementioned communication overhead. But, if you have a task that takes many minutes or more on a single computer, then distribution becomes more promising. 

- Truthfully, hobbyist clusters have become less compelling with the advent of high core count, high clock speed CPUs, as well as GPU task offload. A computer with a sixteen core CPU and 32GB of RAM is going to cost you about 1500 bucks for a reputable, decent example. The same or greater specs can be found in a handful of mini or SFF computers for perhaps 300 bucks. 

As of today, 2/11/26, my work computer has 32gb of RAM and an i7-12700H (14 cores, six of which are for performance, 2.3 Ghz). It was probably two thousand dollars new (granted, it also has discrete Nvidia graphics..). My four node Lenovo cluster (M710q's, four core, 2.4ghz with 8GB ram each) was able to beat it in some prime number tasks - and those Lenovos were sixty dollars a piece. The point being, there's a lot of variability here. 

- Hobbying aside, MPI is still the de-facto spec for large-scale distributed computing and supercomputing. If that's a career path you are interested in, there's nothing betting than getting your hands dirty and working with it directly. If all you want is experience with MPI, then you can buy five twenty dollar raspberry pi's and get cracking. 