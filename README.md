# MIND
MIND: Microstructure INverse Design


conda create -n mind python=3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install click tqdm
pip install pyvista trimesh
pip install scikit-image linear_operator
pip install --no-index torch_sparse -f https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_sparse-0.6.17%2Bpt113cu116-cp310-cp310-linux_x86_64.whl
pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp310-cp310-linux_x86_64.whl
conda install numpy=1.24




conda create -n mind2 python=3.11
conda install numpy=1.26.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install click tqdm pytorch-sparse -c pyg
pip install scikit-image linear_operator pyvista trimes


conda create -n mind2 python=3.11
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install click tqdm pytorch-sparse -c pyg
pip install scikit-image linear_operator pyvista trimesh psutil tensorboardX
