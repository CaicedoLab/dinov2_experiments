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






class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):  #StateDictType.LOCAL_STATE_DICT
            data["model"] = self.model.state_dict()

        # data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT): #StateDictType.LOCAL_STATE_DICT
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler
