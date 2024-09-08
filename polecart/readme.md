
# Setup
Ensure you are using poetry 1.3.2 and python 311.
Download the repo, cd into polecart and run poetry install.


# Usage
To train a model from scratch, run:

```
poetry run python main.py --train
```

Models will be saved every 1000 iterations (configurable in params.py), or when you Ctrl+C out. Models are saved under the saved_models directory,
with the filename format:

```
<datetime>_<iterations>_final_policy_net.pth
```
Models can be loaded back in to carry on training using:

```
poetry run python main.py --train --load <filename>
```

You can emit the --train if you want to see the environment visualisation.

Finally, there is a --report flag which will output training analysis data every 100 iterations. Useful for making code changes to ensure nothing
has gone terribly amiss:

```
poetry run python main.py --train --report
```