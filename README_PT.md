# PyTorch NVS Renderer

| [English](README.md) | Português |

Um renderizador OpenGL para modelos PyTorch de 3D Novel View Synthesis (NVS).

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
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

Faça a build e rode:

```bash
./run.sh ./render.py render_lvsm
```

Os controles são WASD para frente, esquerda, trás, direita, ctrl esquerdo/espaço para baixo/cima e mouse para movimentação de câmera.
Pressione ESC uma vez para destravar o mouse e pressione duas para fechar.

### Usando com outros modelos

#### Como um módulo

Para renderizar usando outro modelo, importe e use a função `render_model` com o seguinte formato:

```py
render_model(n_frames, initial_T, render, device, render_resolution, window_resolution=(800, 800))
```

Onde:

- `n_frames: int`: Número de frames na cena (deve ser 1 no caso de cenas estáticas)
- `initial_T: tensor`: Matriz 4x4 de transformação da câmera inicial
- `render(T: tensor, frame_index: int) -> I: tensor`: Uma função que recebe a matriz 4x4 de transformação de câmera e o índice do frame atual (que vai ser sempre zero em NVS estática) e retorna a imagem renderizada (shape `(C=3, H, W)`) naquela posição
- `device: str`: Qual dispositivo usar (deve ser um dispositivo CUDA)
- `render_resolution: (int, int)`: Qual resolução usar para renderizar imagens (deve ter o mesmo shape que a saída de `render(T)`)
- `window_resolution: (int, int)`: (opcional) Qual resolução usar para a janela

#### Como um script

Crie um módulo similar ao `render_lvsm.py` que exporta todos os parâmetros descritos na seção anterior.

Então faça as configurações usuais, faça a build e rode:

```bash
python render.py --module <nome_do_seu_modulo>
```
