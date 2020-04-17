Automatic Retinal Vessel Segmentation 2
===================================
The final accuracy: 0.946490466594696 (61 epoch).

Limited by the lack of GPU, I use the CPU version of the Pyorch, which makes the training process slow (around 40s for each batch, 40min for the whole training process).

Install:

    The project is based on the Python 3.7 (other version of Python may work correctly).
    pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirement.txt

Usage:

    Training process:
        python train_dl.py
    Testing process:
        python tset_dl.py