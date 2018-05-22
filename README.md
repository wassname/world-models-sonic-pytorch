# world-models-pytorch

My attempt to implement an unsupervised dynamics model with ideas from three papers
- ["World Models"](https://arxiv.org/abs/1803.10122)
- ["MERLIN "or "Unsupervised Predictive Memory in a Goal-Directed Agent"](https://arxiv.org/abs/1803.10760 )
- ["Decoupling Dynamics and Reward for Transfer Learning"](https://arxiv.org/abs/1804.10689)

- ["
Learning and Querying Fast Generative Models for Reinforcement Learning"](https://arxiv.org/abs/1802.03006)
- ["
Probing Physics Knowledge Using Tools from Developmental Psychology"](https://arxiv.org/abs/1804.01128)
- ["
Machine Theory of Mind"](https://arxiv.org/abs/1802.07740)

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
    - [x] work out loss weights
- [x] make sure the rnn loss should be -ve
- [x] make sure rnn loss is using all parts of the sequence
- [x] optimize seq len vs batch of rnn - not enougth gpu ram
- [x] maybe do skip frames - it's already doing 4
- [x] Implement a controller, which uses the dynamics model in the value network (like in MERLIN)

- [ ] latent vims in VAE
    - [x] try 1024z
    - [x] try 512z
    - [x] try 256z - nope
    - [x] try 128z - nope

- [ ] make video of ppo training
    - [ ] save bk2?
    - [ ] matplotlib
    - [ ] save pngs
    - [ ] moviepy https://stackoverflow.com/questions/36401912/is-it-possible-to-generate-a-gif-animation-using-pillow-library-only
- [ ] make video of ppo training with world model decodings
- [ ] make slides


- [ ] discrete actions
    - [x] code
    - [x] gather data
    - [ ] train world model
    - [ ] train ppo

- [ ] why is pytorch running out of ram the second time I run a closed function?
    - [x] cuda free mem doesn't help
    - [ ] what about pin memory?
    - [x] summary after wards helps with 200mb

- [ ] get deep_rl working on ami deep learning. Probobly just need to forward -X of install fake screen


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

## Hyperparameters

### Loss weights

`02_train_dynamics model.ipynb` uses joint training for the VAE, forward, and inverse models (https://arxiv.org/abs/1803.10122). This introduces a few new hyperperameters, but at of 20180520 there is no information on how to set these. The parameters are lambda_vae and lambda_finv.

We want changes to be within an order of magnitude, and we preffer loss VAE to be optimised preferentially, then mdnrnn, then finv. So we want to set it so that loss_vae is large.

For example, if the mdnn is optimised over the VAE, the VAE will learn to output blank images, which the mdnrnn will predict with perfect accuracy. Likewise if the finv is optimised preferentially, the model will only learn to encode the actions in blank images. There are unsatisying local minima.

To set them, you should run for a few epochs with them set to 1, then record the three components of the loss. For example you might get loss_vae=20,000, loss_mdnrnn=3, loss_finv=3. In this case I would set lambda_vae=1/1,00, and the other to one. Keep and eye on the balance between them and make sure they don't get too unbalanced, eventually my unbalanced losses were around loss_vae=2000, loss_mdnrnn=-2, loss_finv=0.1. This means the loss reduction of each was 1800, 5, and 2.5, and the balances loss reductions were 18, 5, and 2.9. All values within an order of mangitude and in an order which follows our preferences.

### Learning rate

Other hyperparamers can sometimes needs to be tweaked. A small learning rate may be needed to initially train the VAE, say 1e-5. Then a higher one may be needed to get the MDRNN to convert, say 3e-4.

Overall it can take quite a small learning rate to train multiple network simulataneusly without being to high on any of them.
