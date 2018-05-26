FROM openai/retro-agent:pytorch-0.4

# Linux deps
RUN apt-get update && \
    apt-get install -y libgtk2.0-dev zlib1g-dev cmake && \
    rm -rf /var/lib/apt/lists/*

# python deps
RUN . ~/venv/bin/activate && \
    pip install gym-retro==0.5.4 git+https://github.com/ShangtongZhang/DeepRL.git@7066fb8e89e9b7e2f029349cdf3fc1d225c0f933#egg=deep_rl

ADD world_models_sonic world_models_sonic
ADD data data
ADD 05_test.py .

CMD ["python", "-u", "/root/compo/05_test.py"]
