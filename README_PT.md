# PyTorch NVS Renderer

| [English](README.md) | Português |

Um renderizador OpenGL para modelos PyTorch de 3D Novel View Synthesis (NVS) que recebe tensores CUDA e renderiza eles diretamente.

Usa a API Python.h para integrar o código Python com o código OpenGL C++.

Por enquanto consegue renderizar cenas do [LVSM](https://github.com/Haian-Jin/LVSM).

Alguns exemplos:

https://github.com/user-attachments/assets/16de9309-e82a-4c30-a4e0-e8285eb4e954

https://github.com/user-attachments/assets/7071a7dc-cba4-410b-ab48-004afeadcf46

## Uso

Clone esse repositório:

```bash
git clone --recursive https://github.com/gammag4/pytorch_nvs_renderer
```

Siga as instruções para configurar o LVSM para inferência [daqui](https://github.com/gammag4/LVSM) (use essa versão porque o código original não é compatível com esse).

Instale as bibliotecas necessárias:

```bash
sudo apt install libglfw3 libglfw3-dev libglm-dev
```

Crie o ambiente conda:

```bash
conda create -n nvs_renderer python=3.13
conda activate nvs_renderer
pip install -r requirements.txt
```

Faça a build e rode:

```bash
./run.sh ./render.py render_lvsm
```

Os controles são WASD para frente, esquerda, trás, direita, ctrl esquerdo/espaço para baixo/cima e mouse para movimentação de câmera.
Pressione ESC uma vez para destravar o mouse e pressione duas para fechar.

## Adicionando modelos

Crie um script similar ao render_lvsm.py que expõe:

- `device (str)`: O dispositivo CUDA usado para renderização
- `initial_cam_state`: Tupla `(x, y, z, rotX, rotY)` com posição e rotação em x e y iniciais da câmera
- `render(T: torch.Tensor): torch.Tensor`: Função que recebe o tensor de transformação de câmera 4x4 e retorna a visão com shape `(3, h, w)` renderizada pelo modelo naquela pose

Então faça a configuração usual, faça a build e rode:

```bash
./run.sh ./render.py <your_script_name>
```
