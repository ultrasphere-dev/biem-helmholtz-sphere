
# GITHUB_TOKEN for readonly access of private repositories
github_readonly_pat:="${MIYABI_GITHUB_TOKEN}"

# TSUBAME settings
# User name
user:="${MIYABI_USER}"
# Group name
group:="${MIYABI_GROUP}"
# The remote server to connect to
remote:=user + "@miyabi-g.jcahpc.jp"
# Your directory in the group drive
group_dir:="/work" / group / user
# The path to the script to be executed
script_path:="miyabi.sh"

default: ssh

# connect (group drive)
ssh:
    ssh -t {{remote}} "cd {{group_dir}}; bash --login"

# run the script (group drive)
run:
    # scp {{script_path}} {{remote}}:{{group_dir}}
    ssh -t {{remote}} "cd {{group_dir}}; bash -l -c 'qsub -W group_list={{group}} $(basename {{script_path}})'"
