
# GITHUB_TOKEN for readonly access of private repositories
github_readonly_pat:="${GSIC_GITHUB_TOKEN}"

# TSUBAME settings
# User name
user:="${GSIC_USER}"
# Group name
group:="${GSIC_GROUP}"
# The remote server to connect to
remote:=user + "@login2.t4.gsic.titech.ac.jp"
# Your directory in the group drive
group_dir:="/gs" / "fs" / group / user
# The path to the script to be executed
script_path:="tsubame.sh"

default: ssh

# replace git url to access private repository on GitHub
init:
    ssh {{remote}} git config --global url."https://{{github_readonly_pat}}@github.com/".insteadOf ssh://git@github.com/

# connect (group drive)
ssh:
	ssh -t {{remote}} "cd {{group_dir}}; bash --login"

# run the script (group drive)
run:
    scp {{script_path}} {{remote}}:{{group_dir}}
    ssh -t {{remote}} "cd {{group_dir}}; bash -l -c 'qsub -g {{group}} $(basename {{script_path}})'"

get:
    rsync -a {{remote}}:{{group_dir}}/biem-helmholtz-sphere/ep/* ep
