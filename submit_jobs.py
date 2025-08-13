import subprocess

submit_script = "submit_job.sh"

scripts = ["SDI_SANA_nerf.py", "samp_SANA_nerf.py"]

exp_name = "exp_2"
n_opts = [50, 200]
lrs = [1e-2, 1e-3, 1e-4]
n_fid = 10

def run(cmd):
    print("Submitting:", " ".join(map(str, cmd)))
    subprocess.run(cmd)

for script in scripts:
    # ---------- Common: our GN/CG variant ----------
    for n_opt in n_opts:
        cmd = [
            "sbatch", submit_script, script,
            "--use-cg",
            "--n-opt", str(n_opt),
            "--n-fid", str(n_fid),
            "--exp-folder", exp_name,
        ]
        run(cmd)

    # ---------- Common: Adam variant ----------
    for n_opt in n_opts:
        for lr in lrs:
            cmd = [
                "sbatch", submit_script, script,
                "--n-opt", str(n_opt),
                "--lr", str(lr),
                "--n-fid", str(n_fid),
                "--exp-folder", exp_name,
            ]
            run(cmd)

    # ---------- Extra grid only for samp_SANA_nerf.py ----------
    if script == "samp_SANA_nerf.py":
        # Case 1: use-ours (default) is already covered above (CG + Adam)

        # Case 2: WITHOUT use-ours -> iterate lr × n_opt (Adam-style)
        for n_opt in n_opts:
            for lr in lrs:
                cmd = [
                    "sbatch", submit_script, script,
                    "--disable-ours",          # <— turn off the default
                    "--n-opt", str(n_opt),
                    "--lr", str(lr),
                    "--n-fid", str(n_fid),
                    "--exp-folder", exp_name,
                ]
                run(cmd)
