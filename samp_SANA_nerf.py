import os
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn.functional as F
import threestudio
import matplotlib.pyplot as plt
import time
from torch.utils._pytree import tree_flatten
from nerf import NeRF2D, forward_with_params, get_image_coords, compute_rhs_b, cg_solve_tuple, matvec_A
from torch.func import functional_call, vjp, jvp
import csv

debug=False


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def main(exp_folder="",
         prompt="bagel filled with cream cheese and lox",
         use_ours=True,
         use_CG=True,
         t_n_steps=40,
         n_presteps=200,
         n_opt=200,
         n_fid=1000,
         lr=1e-3):

    if use_ours:
        optimizer_str = "ours_CG" if use_CG else "ours_Adam"
    else:
        optimizer_str = "Adam"
    exp_name = f"./{exp_folder}/sample_SANA_NeRF_{optimizer_str}_n_presteps_{n_presteps}_n_opt_{n_opt}"
    if "Adam" in optimizer_str:
        exp_name += f"_lr_{lr}"
    print(f"Experiment name: {exp_name}")
    debug_folder = os.path.join(exp_name, "debug")
    os.makedirs(debug_folder, exist_ok=True)

    all_mse = {}  # key: seed, value: list of MSEs for each step
    config = {
        'seed': 0,
        'scheduler': None,
        'mode': 'latent',
        'prompt_processor_type': 'sana-prompt-processor',
        'prompt_processor': {
            'pretrained_model_name_or_path': "Efficient-Large-Model/Sana_600M_512px_diffusers",
            'prompt': prompt,
            'use_perp_neg': False,
            'spawn' : False

        },
        'guidance_type': 'sana-guidance',
        'guidance': {
            'pretrained_model_name_or_path': "Efficient-Large-Model/Sana_600M_512px_diffusers",
            'guidance_scale': 4.5,
            'weighting_strategy': "sds",
            'min_step_percent': 0.02,
            'max_step_percent': 0.98,
            'grad_clip': None,
            'num_inference_steps': t_n_steps,
            # SDI parameters
            # 'enable_sdi': True,
            # 'inversion_guidance_scale': -7.5,
            # 'inversion_n_steps': 10,
            # 'inversion_eta': 0.3,
            # 't_anneal': True
        },
        'latent': {
            'width': 16,
            'height': 16,
        },
    }

    batch = {
        'elevation': torch.Tensor([0]),
        'azimuth': torch.Tensor([0]),
        'camera_distances': torch.Tensor([1]),
    }


    guidance = threestudio.find(config['guidance_type'])(config['guidance'])
    prompt_processor = threestudio.find(config['prompt_processor_type'])(config['prompt_processor'])

    B = 1

    w, h = config['latent']['width'], config['latent']['height']
    coords = get_image_coords(h, w, device='cuda')

    start_time = time.time()  # Start timing


    for seed in tqdm(range(n_fid)):
        img_path = f"{exp_name}/{seed}.png"
        # if os.path.exists(img_path):
        #     continue
        config['seed'] = seed
        mse_steps = []  # store MSE per step for this seed

        seed_everything(config['seed'])

        guidance.reset_guidance()

        # Initialization
        t_tensor = torch.tensor(B * [999], device=guidance.device)
        init_noise = torch.randn(B, 32, h, w, device=guidance.device)
        # init_noise_rgb = guidance.decode_latents(init_noise, output_type="pt").to(torch.float32)
        # guidance.decode_latents(init_noise, output_type="pil")[0].save(os.path.join(exp_name, f"init_noise.png"))

        model = NeRF2D(D=1, W=128).to(guidance.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

        for pretrain_step in range(n_presteps):
            latent = model.forward(coords, height=h, width=w)
            loss_pretrain = F.mse_loss(latent, init_noise)

            loss_pretrain.backward()
            optimizer.step()
            optimizer.zero_grad()

        # with torch.no_grad():
        #     latent = model.forward(coords, height=h, width=w)
        #     loss_pretrain = F.mse_loss(latent, init_noise)
        #     print(f"Final Latent Loss: {loss_pretrain}")

        #     save_path = os.path.join(exp_name, f"init_noise_in_model.png")
        #     guidance.decode_latents(latent, output_type="pil")[0].save(save_path)


        flat_params, param_spec = tree_flatten(dict(model.named_parameters()))
        flat_params = [p.detach().requires_grad_(True) for p in flat_params]

        total_params = sum(p.numel() for p in flat_params)
        # print(f"Total number of parameters: {total_params}")
        forward_fun = forward_with_params(param_spec, model, coords, h, w)
        u_solution = list(torch.zeros_like(x) for x in flat_params)

        def loss_fn(u_list):
            # u_list: list of tensors (same shape as flat_params)
            _, Ju = jvp(forward_fun, (flat_params,), (u_list,))
            return F.mse_loss(Ju, flow_pred)
        # total_steps = (t_n_steps-1) * n_opt

        for diffusion_step in range(t_n_steps):
                with torch.no_grad():
                    # Step 1: Render current latent
                    if use_ours:
                        latent = forward_fun(flat_params)
                    else:
                        latent = model.forward(coords, height=h, width=w)
                    
                    # Step 2: Get predicted flow in latent space
                    batch_size = latent.shape[0]
                    t_tensor = guidance.timesteps[guidance.sampling_index].repeat(batch_size)
                    flow_pred, _, _ = guidance.predict_flow(
                        latent,
                        t_tensor,
                        prompt_processor(),
                        **batch,
                        guidance_scale=config['guidance']['guidance_scale']
                    ) # v ~ (eps - z_0) -> z_next = z_prev - flow_pred * delta_t

                sigma_t = guidance.scheduler.sigmas[guidance.sampling_index]
                sigma_next = guidance.sigmas[guidance.sampling_index + 1]
                delta_t = sigma_t - sigma_next

                target_latent = latent - flow_pred * delta_t

                if use_ours:
                    if use_CG:
                        with torch.no_grad():
                            rhs_b = compute_rhs_b(flat_params, forward_fun, flow_pred)

                            u_solution, cg_mse = cg_solve_tuple(
                                lambda u: matvec_A(u, flat_params, forward_fun, lambda_reg=1e-3),
                                rhs_b,
                                u_solution,
                                max_iter=n_opt,
                                tol=1e-6
                            ) # solves (J^T J) u = J^T b
                            # print(mse)

                    else:
                        u_solution = [u.detach().clone().requires_grad_(True) for u in u_solution]
                        optimizer = torch.optim.Adam(u_solution, lr=lr)
                        for i in range(n_opt):
                            optimizer.zero_grad()
                            loss = loss_fn(u_solution)
                            loss.backward()
                            optimizer.step()
                        
                        # Store solution for next step
                        u_solution = [u.detach().clone() for u in u_solution]

                    with torch.no_grad():
                        flat_params = [p - delta_t * u for p, u in zip(flat_params, u_solution)]
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    for i in range(n_opt):
                        output = model.forward(coords, height=h, width=w)
                        loss = F.mse_loss(output, target_latent)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    

                # Compute Ju
                # _, Ju_solution = jvp(lambda p: forward_with_params(p, param_spec, model, coords, h, w),(flat_params,), (u_solution,))
                
                # Compare with v (flow_pred)
                # residual = Ju_solution - flow_pred
                # residual_norm_sq = torch.sum(residual ** 2)
                # residual_mse = residual_norm_sq / residual.numel()
                # print(f"(1/N)||Ju - v||^2 (MSE): {residual_mse.item():.6f}")
                with torch.no_grad():
                    if use_ours:
                        latent = forward_fun(flat_params)
                    else:
                        latent = model.forward(coords, height=h, width=w)
                    mse_val = F.mse_loss(latent, target_latent).item()
                    mse_steps.append(mse_val)

                if debug or seed == 0:
                    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
                    ax[0].imshow(guidance.decode_latents(latent).squeeze())
                    ax[1].imshow(guidance.decode_latents(flow_pred).squeeze())
                    ax[0].set_title('Current Image')
                    ax[1].set_title('Target Velocity')
                    ax[0].axis('off')
                    ax[1].axis('off')
                    save_path = os.path.join(debug_folder, f"step_{diffusion_step:04d}.png")
                    fig.savefig(save_path)
                    plt.close(fig)  # important to avoid memory leaks
                    print(f"Step {diffusion_step}")

                

                # save_path = os.path.join(exp_name, f"step_{diffusion_step:04d}.png")
                # guidance.decode_latents(latent, output_type="pil")[0].save(save_path)

                guidance.update_step()

        with torch.no_grad():
            if use_ours:
                latent = forward_fun(flat_params)
            else:
                latent = model.forward(coords, height=h, width=w)
        guidance.decode_latents(latent, output_type="pil")[0].save(img_path)
        all_mse[seed] = mse_steps

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    # Save to file
    with open(f"{exp_name}/total_time.txt", "w") as f:
        f.write(f"{total_time:.2f} seconds\n")
        csv_path = os.path.join(exp_name, "mse_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["seed"] + [f"step_{i}" for i in range(t_n_steps)]
        writer.writerow(header)
        for seed, mse_list in sorted(all_mse.items()):
            writer.writerow([seed] + mse_list)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-folder", type=str, default="", help="Experiment folder")
    parser.add_argument("--prompt", type=str, default="bagel filled with cream cheese and lox")
    parser.add_argument("--disable-ours", action="store_true", default=False, help="Disable our solver")
    parser.add_argument("--use-cg", action="store_true", default=False, help="Enable CG solver instead of Adam")
    parser.add_argument("--t-n-steps", type=int, default=40)
    parser.add_argument("--n-presteps", type=int, default=500)
    parser.add_argument("--n-opt", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n-fid", type=int, default=10, help="How many seeds/images to generate")

    args = parser.parse_args()

    main(
        exp_folder=args.exp_folder,
        prompt=args.prompt,
        use_ours= not args.disable_ours,
        use_CG=args.use_cg,
        t_n_steps=args.t_n_steps,
        n_presteps=args.n_presteps,
        n_opt=args.n_opt,
        n_fid=args.n_fid,
        lr=args.lr
    )