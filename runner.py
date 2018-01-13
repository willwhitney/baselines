import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
local = '--local' in sys.argv
detach = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

basename = "trpo_16x16_test"
# grids = [
#     {
#         "curriculum": [1],  # advance to step k+1 when reward is >= 1 - k/35
#     }
# ]
grids = [
    {
        "curriculum": [0, 1],  # advance to step k+1 when reward is >= 1 - k/35
        "walldeath": [0, 1],  # episode ends if the agent runs into a wall
        "env-size": [8],
        "max-kl": [0.01, 0.001],
        "lam": [1.0, 0.98],
        "seed": [0, 1, 2],
    }
]

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if dry_run:
    print("NOT starting jobs:")
else:
    print("Starting jobs:")

merged_grid = {}
for grid in grids:
    for key in grid:
        merged_grid[key] = [] if key not in merged_grid else merged_grid[key]
        merged_grid[key] += grid[key]

varying_keys = {key for key in merged_grid if len(merged_grid[key]) > 1}

for job in jobs:
    jobname = basename
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                flagstring = flagstring + " --" + flag
                if flag in varying_keys:
                    jobname = jobname + "_" + flag + str(job[flag])
            else:
                print("WARNING: Excluding 'False' flag " + flag)
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                flagstring = flagstring + " --" + \
                    flag + " " + str(network_location)
            else:
                flagstring = flagstring + " --" + flag + " " + \
                    networks_prefix + "/" + str(imported_network_name)
            if flag in varying_keys:
                jobname = jobname + "_" + flag + str(imported_network_name)
        else:
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
            if flag in varying_keys:
                jobname = jobname + "_" + flag + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = ("mpirun -n 8 --oversubscribe "
#     jobcommand = (""
                  "python -m baselines.trpo_mpi.run_gridworld") + flagstring
    print(jobcommand)

    if local and not dry_run:
        if detach:
            os.system(jobcommand + ' 2> slurm_logs/' + jobname +
                      '.err 1> slurm_logs/' + jobname + '.out &')
        else:
            os.system(jobcommand)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" +
                            jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write(jobcommand + "\n")


        if not dry_run:
            os.system((
                "sbatch --nodes 1 -c 8 --mem=32000 "
                "--time=7-00:00:00 slurm_scripts/" + jobname + ".slurm &"))
