Last login: Wed May  8 19:59:06 on console

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
Adityas-MacBook-Pro-2:~ adityapillai$ ssh appillai@ap2002.chtc.wisc.edu
(appillai@ap2002.chtc.wisc.edu) Password: 
(appillai@ap2002.chtc.wisc.edu) Duo two-factor login for appillai

Enter a passcode or select one of the following options:

 1. Duo Push to XXX-XXX-6979

Passcode or option (1-1): 1
Success. Logging you in...
Success. Logging you in...
Last login: Wed May  8 21:35:59 2024 from 10.130.176.99
_____________________________________________________________________
 #####  #     # #######  #####  Issues?  Email chtc@cs.wisc.edu
#     # #     #    #    #     # Unauthorized use prohibited by:
#       #     #    #    #       WI Statutes: s. 947.0125
#       #######    #    #       U.S. Code: 18 USC 1030
#       #     #    #    #       U.S. Code: 18 USC 2510-2522
#     # #     #    #    #     # U.S. Code: 18 USC 2701-2712
 #####  #     #    #     #####  U.S. Code: 18 USC § 1831
For off campus ssh access use https://www.doit.wisc.edu/network/vpn/
_____________________________________________________________________

           Virtual office hours are available twice a week:
 Tuesdays, 10:30am - 12pm and Thursdays, 3:00 - 4:30pm (Central time)
           Join via this link: go.wisc.edu/chtc-officehours
        Sign in via this link: go.wisc.edu/chtc-officehours-signin
Filesystem quota report
Storage           Used (GB)    Limit (GB)    Files (#)    File Cap (#)    Quota (%)
--------------  -----------  ------------  -----------  --------------  -----------
/home/appillai         0.84            50         3073               0         1.67

[appillai@ap2002 ~]$ cd dinoExperiments/
[appillai@ap2002 dinoExperiments]$ ls
trained
[appillai@ap2002 dinoExperiments]$ cd trained/
[appillai@ap2002 trained]$ ls
config.yaml        datasetinstall.sub  gpu-chtc_1567674.log  gpu-chtc_1569705.log  gpu-chtc_1569884.log  gpu-chtc_1572552.log  gpu-chtc_1572988.log  __init__.py   training.sh   train.py
datasetinstall.sh  docker_stderror     gpu-chtc_1569705.err  gpu-chtc_1569705.out  gpu-chtc_1569885.log  gpu-chtc_1572840.log  gpu-chtc_1573440.log  testmultigpu  training.sub
[appillai@ap2002 trained]$ cd testmultigpu/
[appillai@ap2002 testmultigpu]$ s
-bash: s: command not found
[appillai@ap2002 testmultigpu]$ ls
config.yaml          docker_stderror  gpu.sub                 __init__.py         log        rank0_contact  train.dag.condor.sub  train.dag.dagman.out  train.dag.lib.out  train.dag.nodes.log  train.dag.rescue001.old  train.py
debug-cli.25805.log  err.rank0        gpuworker.sub.template  injectRank0Name.sh  out.rank0  train.dag      train.dag.dagman.log  train.dag.lib.err     train.dag.metrics  train.dag.rescue001  trainingScript.sh
[appillai@ap2002 testmultigpu]$ vim err.rank0
[appillai@ap2002 testmultigpu]$ ls
config.yaml          docker_stderror  gpu.sub                 __init__.py         log        rank0_contact  train.dag.condor.sub  train.dag.dagman.out  train.dag.lib.out  train.dag.nodes.log  train.dag.rescue001.old  train.py
debug-cli.25805.log  err.rank0        gpuworker.sub.template  injectRank0Name.sh  out.rank0  train.dag      train.dag.dagman.log  train.dag.lib.err     train.dag.metrics  train.dag.rescue001  trainingScript.sh
[appillai@ap2002 testmultigpu]$ cd ..
[appillai@ap2002 trained]$ ls
config.yaml        datasetinstall.sub  gpu-chtc_1567674.log  gpu-chtc_1569705.log  gpu-chtc_1569884.log  gpu-chtc_1572552.log  gpu-chtc_1572988.log  __init__.py   training.sh   train.py
datasetinstall.sh  docker_stderror     gpu-chtc_1569705.err  gpu-chtc_1569705.out  gpu-chtc_1569885.log  gpu-chtc_1572840.log  gpu-chtc_1573440.log  testmultigpu  training.sub
[appillai@ap2002 trained]$ vim train.py
[appillai@ap2002 trained]$ vim training.sub
[appillai@ap2002 trained]$ vim training.sh
[appillai@ap2002 trained]$ vim config.yaml 
[appillai@ap2002 trained]$ vim __init__.py 
[appillai@ap2002 trained]$ vim training.sh
[appillai@ap2002 trained]$ vim training.sub
[appillai@ap2002 trained]$ vim training.sub
[appillai@ap2002 trained]$ cd testmultigpu/
[appillai@ap2002 testmultigpu]$ ls
config.yaml          docker_stderror  gpu.sub                 __init__.py         log        rank0_contact  train.dag.condor.sub  train.dag.dagman.out  train.dag.lib.out  train.dag.nodes.log  train.dag.rescue001.old  train.py
debug-cli.25805.log  err.rank0        gpuworker.sub.template  injectRank0Name.sh  out.rank0  train.dag      train.dag.dagman.log  train.dag.lib.err     train.dag.metrics  train.dag.rescue001  trainingScript.sh
[appillai@ap2002 testmultigpu]$ vim __init
[appillai@ap2002 testmultigpu]$ vim __init__.py 
[appillai@ap2002 testmultigpu]$ vim config.yaml 
[appillai@ap2002 testmultigpu]$ vim trainingScript.sh 
[appillai@ap2002 testmultigpu]$ vim train.py

                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        #wandb logging
        wandb.log({"loss": losses_reduced})
        wandb.log({"learning rate": lr})


        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@record
def main(args):
    #wandb.login()

    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
