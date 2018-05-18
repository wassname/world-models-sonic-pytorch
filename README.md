# world-models-pytorch

My attempt to implement an unsupervised dynamics model with ideas from three papers
- ["World Models"](https://arxiv.org/abs/1803.10760)
- ["MERLIN "or "Unsupervised Predictive Memory in a Goal-Directed Agent"](https://arxiv.org/abs/1804.10689)
- ["Decoupling Dynamics and Reward for Transfer Learning"](https://arxiv.org/abs/1803.10122)


## The work is in progress

- implementations, which I got code and ideas from
    - https://github.com/AppliedDataSciencePartners/WorldModels
    - https://github.com/JunhongXu/world-models-pytorch
    - https://github.com/goolulusaurs/WorldModels
    - https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb

# Competition links

links:
- https://discordapp.com/channels/427882398005329921/

# Plans

TODOs

- [x] Implement MDN-RNN
- [x] Implement VAE
- [x] Training for VAE
    - [x] Random rollouts to create a dataset
    - [x] Training function
- [x] Training for MDN-RNN
- [ ] Evolution strategy for training Controller

- [x] maybe use full size frames?
- [x] skip connections? hard to do in VAE's, but I used inception bocks
- [x] dropout, batchnorm, leakyrelu. Inception blocks.
    - [x] dropout hurt performanc and seems to absent from most VAE code
- [x] try smoothl1? doesn't seem to help although this might be the KLD reconstructon loss balance. Plus it's that same as L2 when loss>1.

- [x] make a module containing all 3 including inverse model from https://arxiv.org/abs/1804.10689
    - that way they can use the same optimizer
- [x] make inverse model `a = F(z', z)` should be an RNN with latent_dim* 2 in and action_dim out. So similar to the forward
    - [x] should it be statistical, a mdn?
    with
        latent d = 256. Both the LSTM-D and LSTM-R have a hidden
        layer with 128 units each. The Inverse model,finv, consists
        of a linear layer of size 64 with ReLU non-linearity followed
        by an output layer of size 4 with the softmax activation
        defining a probability over actions. T
- [x] do triple training
- [ ] make sure the rnn loss should be -ve
- [ ] make sure rnn loss is using all parts of the sequence
- [ ] optimize seq len vs batch of rnn
- [ ] maybe do skip frames
- [ ] Implement a controller, which uses the dynamics model in the value network (like in MERLIN)

- [ ] latent vims in VAE
    - [x] try 1024z
    - [x] try 512z
    - [ ] try 256z
    - [ ] try 128z

## Setup

- requirements (see `./requirements/requirements.txt`)
    - [openai/retro](https://github.com/openai/retro)
    - [openai/gym_remote (from retro-contest)](https://github.com/openai/retro-contest)
    - [openai/retro-baselines](https://github.com/openai/retro-baselines/blob/master/agents/ppo2.docker)
    

- roms: you need the ROM files for the Sonic games
    - install from steam
        - [Sonic The Hedgehog](http://store.steampowered.com/app/71113/Sonic_The_Hedgehog/)
        - [Sonic The Hedgehog 2](http://store.steampowered.com/app/71163/Sonic_The_Hedgehog_2/)
        - [Sonic 3 & Knuckles](http://store.steampowered.com/app/71162/Sonic_3__Knuckles/)
        - so download rom files run [this](https://github.com/openai/retro/blob/master/retro/scripts/import_sega_classics.py) then find the rom files in path given by `retro.data_path()`
    - do not get from these links (unless you already have the license to these specific versions and that gives you the rights in your country)
        - [Sonic The Hedgehog (Japan, Korea).md](http://www.completeroms.com/dl/sega-genesis/sonic-the-hedgehog-japan-korea/151020)
        - [Sonic The Hedgehog 2 (World) (Rev A).md](http://www.completeroms.com/dl/game-gear/sonic-the-hedgehog-2-u-/7772)
        - [Sonic & Knuckles + Sonic The Hedgehog 3 (USA).md](http://www.completeroms.com/dl/sega-genesis/sonic-and-knuckles-sonic-3-jue-/1824)
        
### Running

I have included the pretrained models in releases

- 01_gather_dynamics training_data.py
- 02_train_dynamics model.ipynb
- 03_train_controller.ipynb
