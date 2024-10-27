# MeetingCoach
Face&amp;Motion regognition And Speaker Diarization for Meeting Feedback

- diarization -> hugging face pyannote/speaker-diarization-3.1

  - torch 2.5.0+cu124

  - cuda toolkit 12.4

  - cudnn 8.9x

- voice recognize -> google speech to text
- mp4 to wav -> moviepy
- llm sumurize -> openai(Okestro LLM)

[requirements.txt]

```txt
3-1==1.0.0
aiohappyeyeballs==2.4.3
aiohttp==3.10.10
aiosignal==1.3.1
alembic==1.13.3
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.6.2.post1
asteroid-filterbanks==0.4.0
attrs==24.2.0
audioread==3.0.1
cachetools==5.5.0
certifi==2024.8.30
cffi==1.17.1
charset-normalizer==3.4.0
click==8.1.7
colorama==0.4.6
colorlog==6.8.2
contourpy==1.3.0
cycler==0.12.1
decorator==4.4.2
distro==1.9.0
docopt==0.6.2
einops==0.8.0
filelock==3.16.1
fonttools==4.54.1
frozenlist==1.5.0
fsspec==2024.10.0
google-api-core==2.21.0
google-auth==2.35.0
google-cloud-speech==2.28.0
googleapis-common-protos==1.65.0
greenlet==3.1.1
grpcio==1.67.0
grpcio-status==1.67.0
h11==0.14.0
httpcore==1.0.6
httpx==0.27.2
huggingface-hub==0.26.1
HyperPyYAML==1.2.2
idna==3.10
imageio==2.36.0
imageio-ffmpeg==0.5.1
Jinja2==3.1.4
jiter==0.6.1
joblib==1.4.2
julius==0.2.7
kiwisolver==1.4.7
lazy_loader==0.4
librosa==0.10.2.post1
lightning==2.4.0
lightning-utilities==0.11.8
llvmlite==0.43.0
Mako==1.3.6
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.9.2
mdurl==0.1.2
moviepy==1.0.3
mpmath==1.3.0
msgpack==1.1.0
multidict==6.1.0
networkx==3.4.2
numba==0.60.0
numpy==1.26.4
omegaconf==2.3.0
openai==1.52.2
optuna==4.0.0
packaging==24.1
pandas==2.2.3
pillow==11.0.0
platformdirs==4.3.6
pooch==1.8.2
primePy==1.3
proglog==0.1.10
propcache==0.2.0
proto-plus==1.25.0
protobuf==5.28.3
pyannote.audio==3.3.2
pyannote.core==5.0.0
pyannote.database==5.1.0
pyannote.metrics==3.2.1
pyannote.pipeline==3.0.1
pyasn1==0.6.1
pyasn1_modules==0.4.1
pycparser==2.22
pydantic==2.9.2
pydantic_core==2.23.4
pydub==0.25.1
Pygments==2.18.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytorch-lightning==2.4.0
pytorch-metric-learning==2.6.0
pytz==2024.2
PyYAML==6.0.2
regex==2024.9.11
requests==2.32.3
rich==13.9.3
rsa==4.9
ruamel.yaml==0.18.6
ruamel.yaml.clib==0.2.12
safetensors==0.4.5
scikit-learn==1.5.2
scipy==1.14.1
semver==3.0.2
sentencepiece==0.2.0
shellingham==1.5.4
six==1.16.0
sniffio==1.3.1
sortedcontainers==2.4.0
soundfile==0.12.1
soxr==0.5.0.post1
speechbrain==1.0.1
SQLAlchemy==2.0.36
sympy==1.13.1
tabulate==0.9.0
tensorboardX==2.6.2.2
threadpoolctl==3.5.0
tokenizers==0.20.1
torch==2.5.0+cu124
torch-audiomentations==0.11.1
torch_pitch_shift==1.2.5
torchaudio==2.5.0+cu124
torchmetrics==1.5.1
torchvision==0.20.0+cu124
tqdm==4.66.5
transformers==4.46.0
typer==0.12.5
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
yarl==1.16.0

```



