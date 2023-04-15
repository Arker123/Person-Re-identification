
Torchreid is a library for deep-learning person re-identification, written in `PyTorch <https://pytorch.org/>`_ and developed for our ICCV'19 project, `Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>`_.


Code: https://github.com/KaiyangZhou/deep-person-reid.

Documentation: https://kaiyangzhou.github.io/deep-person-reid/.
	

Installation
---------------

Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.


.. code-block:: bash

    # cd to your preferred directory and clone this repo
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop


A unified interface
-----------------------
In "deep-person-reid/scripts/", we provide a unified interface to train and test a model. See "scripts/main.py" and "scripts/default_config.py" for more details. The folder "configs/" contains some predefined configs which you can use as a starting point.

Below we provide an example to train and test `OSNet (Zhou et al. ICCV'19) <https://arxiv.org/abs/1905.00953>`_. Assume :code:`PATH_TO_DATA` is the directory containing reid datasets. The environmental variable :code:`CUDA_VISIBLE_DEVICES` is omitted, which you need to specify if you have a pool of gpus and want to use a specific set of them.

Conventional setting
^^^^^^^^^^^^^^^^^^^^^

----------------------Follow the commands below to run the code------------------------------

To train OSNet on Market1501, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA


The config file sets Market1501 as the default dataset. If you wanna use DukeMTMC-reID, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    -s dukemtmcreid \
    -t dukemtmcreid \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA \
    data.save_dir log/osnet_x1_0_dukemtmcreid_softmax_cosinelr

The code will automatically (download and) load the ImageNet pretrained weights. After the training is done, the model will be saved as "log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250". Under the same folder, you can find the `tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ file. To visualize the learning curves using tensorboard, you can run :code:`tensorboard --logdir=log/osnet_x1_0_market1501_softmax_cosinelr` in the terminal and visit :code:`http://localhost:6006/` in your web browser.

Evaluation is automatically performed at the end of training. To run the test again using the trained model, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --root $PATH_TO_DATA \
    model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
    test.evaluate True


