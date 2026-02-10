a straightforward mpi library for python.



desired tasks to distribute:

1. matmul
2. video encoding
3. file compression
4. csv-> json transforms?
5. prime testing


other needs: 
1. in-script infra configuration
    a. ssh credentials
    b. cores, threads assigned per machine
    c. number of machines assigned
    d. which node is master node and performs orchestration
    **assuming all worker machines are identical in specs
2. per-feature unit tests with both happy and sad paths. 




outlines:

configure_infra = {
    master_node = master name,
    per_node_cores = x,
    per_node_threads = y,
    num_worker_nodes = z,
    time_job = bool (returns how long the job took, to millisecond precision),
    progress_to_terminal = bool (sends info to terminal like "jobs sent", etc)
}
    if per_node_threads is null, do not use multithreading
    master_node cannot be null
    per_node_cores cannot be null
    num_worker_nodes cannot be null