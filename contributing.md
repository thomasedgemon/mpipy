## Contributing

- A new feature should follow the same format as pre-existing ones. It should be structured as a function with only the necessary args. 
- A new feature should be submitted alongside a test. This test should pass before PR is opened. 
- A new feature should rely on the already-instantiated user config. The feature will distribute its job across all specified cores and workers to the greatest degree possible, taking into careful consideration how that job division happens should there be argument variability (like matrix sizes and how the 2d decomposition takes place based on the user-specified number of cores to be used.)
- A new feature should be accompanied by docs.
- A new feature should be one which is extremely parallelizable, and therefore beneficial to run on a cluster. In that vein, it should allow for arbitrarily large inputs/args. Performance increase is obviously going to vary depending on the size of the user's cluster - but the opportunity for performance increase should be apparent. 