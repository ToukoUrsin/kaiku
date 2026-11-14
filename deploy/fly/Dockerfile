# The live demo as one container: 48 simulated robots, the page, the recorder and the trainer.
#   deploy/deploy.sh      (from the repo's root; it also puts the same code on Modal, where Train trains)
# Train uses Modal when the app has MODAL_TOKEN_ID and MODAL_TOKEN_SECRET as secrets, and this machine otherwise.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install numpy==2.5.4 mujoco==4.0.1 pyyaml scipy mink==1.4.0 "qpsolvers[daqp]" aiohttp websockets msgpack pyarrow==26.0.1 modal==1.6.4
WORKDIR /app
COPY kaiku ./kaiku
COPY third_party/mujoco_menagerie/unitree_g1 ./third_party/mujoco_menagerie/unitree_g1
EXPOSE 8080
CMD ["python", "-m", "kaiku.app.run", "--n", "48", "--host", "0.0.0.0", "--port", "8080", \
     "--record", "/data/live1", "--train", "modal", "--train-steps", "30000", "--train-about", "25", "--shared"]
