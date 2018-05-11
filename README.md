# world-models-pytorch
My attempt to implement "World Models" by David Ha and Jurgen Schmidhuber
Link: https://arxiv.org/pdf/1803.10122.pdf

## The work is in progress

- https://github.com/AppliedDataSciencePartners/WorldModels
- https://github.com/JunhongXu/world-models-pytorch
- https://github.com/goolulusaurs/WorldModels

TODOs

- [x] Implement MDN-RNN
- [x] Implement VAE
- [x] Training for VAE
    - [x] Random rollouts to create a dataset
    - [x] Training function
- [ ] Training for MDN-RNN
- [ ] Evolution strategy for training Controller

- [ ] maybe use full size frames?
- [ ] skip connections?
- [x] dropout, batchnorm, leakyrelu
- [ ] try smoothl1?

## Setup

- roms
    - install from steam
        - [Sonic The Hedgehog](http://store.steampowered.com/app/71113/Sonic_The_Hedgehog/)
        - [Sonic The Hedgehog 2](http://store.steampowered.com/app/71163/Sonic_The_Hedgehog_2/)
        - [Sonic 3 & Knuckles](http://store.steampowered.com/app/71162/Sonic_3__Knuckles/)
        - so download rom files run [this](https://github.com/openai/retro/blob/master/retro/scripts/import_sega_classics.py) then find the rom files in path given by `retro.data_path()`
    - do not get from these links unless you already have the license to these specific version (they have the same hashes)
        - [Sonic The Hedgehog (Japan, Korea).md](http://www.completeroms.com/dl/sega-genesis/sonic-the-hedgehog-japan-korea/151020)
        - [Sonic The Hedgehog 2 (World) (Rev A).md](http://www.completeroms.com/dl/game-gear/sonic-the-hedgehog-2-u-/7772)
        - [Sonic & Knuckles + Sonic The Hedgehog 3 (USA).md](http://www.completeroms.com/dl/sega-genesis/sonic-and-knuckles-sonic-3-jue-/1824)

- requirements (see `./requirements/requirements.txt`)
    - [openai/retro](https://github.com/openai/retro)
    - [openai/gym_remote (from retro-contest)](https://github.com/openai/retro-contest)
    - [openai/retro-baselines](https://github.com/openai/retro-baselines/blob/master/agents/ppo2.docker)
