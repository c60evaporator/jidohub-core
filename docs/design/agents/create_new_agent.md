## Agent作成方法

### Dockerfileの作成方法

#### devel用Dockerfileの作成

#### Dockerfile作成時の注意点

- 重みダウンロードはなるべくHuggingFace等を活用したい
- 外部の重みを使用する場合はライセンスに注意

##### `--no-build-isolation`を付けるかどうかの判断

GitHubリポジトリで公開されているAIモデルを用いる場合、インストール手順に`pip install -e .`コマンドが含まれているケースが多いでしょう（基盤モデル系に多い）。このコマンドでは、リポジトリ内の`setup.py`に基づき依存パッケージがインストールされます。

このとき、Dockerコンテナ内にこのAIモデルをインストールする際に`RUN pip install -e .`とすると、**以下の条件**においてビルドがうまく行きません。

- **setup.py内で**`import torch`**している**
- CUDA ExtensionをCppExtensionまたはCUDAExtensionでビルドしている（setup.py内に`from torch.utils.cpp_extension import CUDAExtension`のような記述がある）

この原因として、Dockerファイル内での`RUN pip install -e .`では新しい仮想環境が作成されてその中でビルドが行われる（build isolation）ため、この環境の中にtorchがなく`import torch`が失敗したり、requirements.txt等に基づき隔離環境内に異なるバージョンのtorchがインストールされバージョン違いのビルドが行われることが挙げられます。

**上記条件を満たすときは**、build isolationを無効化にするための`--no-build-isolation`オプションをつけて

```python
RUN pip install --no-build-isolation -e .
```

としてください

##### `TORCH_CUDA_ARCH_LIST`

これはエージェント作成時ではなくエージェントをダウンロードして使用するユーザー側が気にすべき事項です。
インストールにNVIDIA GPU用のC/C++コードコンパイル（nvccビルド）が必要なライブラリは、Dockerコンテナでのビルド時に`TORCH_CUDA_ARCH_LIST`という**GPUアーキテクチャを指定する環境変数**を指定する必要があります。

使用しているGPUの`TORCH_CUDA_ARCH_LIST`は、[こちらのサイト](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)の表の`Compute capability`列を`GeForce`列のGPU名と比較することで特定する事ができます

例えばRTX4900やRTX6000Adaの場合、`Compute capability`列が8.9となっているので、

```python
export TORCH_CUDA_ARCH_LIST=8.9
docker compose build
```

のように`TORCH_CUDA_ARCH_LIST`環境変数をexportしてからビルドすればOKです
