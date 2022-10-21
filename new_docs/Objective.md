# 1. What's the INC objective

In terms of evaluating the status of a specific model during tuning, we should have general objectives. Neural Compressor Objective supports code-free configuration through a yaml file. With built-in objectives, users can compress model with different objectives easily. In special cases, users can also register their own objective classes.

## 1.1 Single objective

The objective supported by Neural Compressor is driven by accuracy. If users want to evaluate a specific model with other objectives, they can realize it with `objective` in a yaml file. Default value for `objective` is `performance`, and the other values are `modelsize` and `footprint`.

## 1.2 Multiple objective

In some cases, users want to use more than one objective to evaluate the status of a specific model and they can realize it with `multi_objectives` in a yaml file. Currently `multi_objectives` supports built-in objectives.

If users use `multi_objectives` to evaluate the status of a model during tuning, Neural Compressor will return a model with the best score of `multi_objectives` and meeting `accuracy_criterion` after tuning ending.

When calculating the weighted score of objectives, Neural Compressor will normalize the results of objectives to [0, 1] one by one first.


# 2. Objective support matrix

Built-in objectives support list:

| Objective    | Usage                                                    |
| :------      | :------                                                  |
| accuracy     | Evaluate the accuracy                                    |
| performance  | Evaluate the inference time                              |
| footprint    | Evaluate the peak size of memory blocks during inference |
| modelSize    | Evaluate the model size                                  |

# 3. Objective API summary

```python
# objective.py in neural_compressor
class Objective(object):
    representation = ''

    def __init__(self):
        """Initialize `Objective` class"""
        ...

    @abstractmethod
    def reset(self):
        """The interface reset objective measuring."""
        ...

    @abstractmethod
    def start(self):
        """The interface start objective measuring."""
        ...

    @abstractmethod
    def end(self):
        """The interface end objective measuring."""
        ...

    @property
    def model(self):
        """Getter of the model object"""
        ...

    @model.setter
    def model(self, model):
        """Setter of the model object"""
        ...

    def result(self, start=None, end=None):
        """
        The interface to get objective measuring result.
        Measurer may sart and end many times. This interface 
        will return the total mean of the result.
        """
        ...

    def result_list(self):
        """
        The interface to get objective measuring result list.
        This interface will return a list of each start-end loop
        measure value.
        """
        ...

    def __str__(self):
        """Get representation"""
        ...
```

# 4. Examples

## 4.1 Config built-in objective in a yaml file

Users can specify a built-in objective as shown below:

```yaml
tuning:
  objective: performance
```

## 4.2 Config multi_objectives in a yaml file

Users can specify built-in multi-objective as shown below:

```yaml
tuning:
  multi_objectives:
    objective: [accuracy, performance]
    higher_is_better: [True, False]
    weight: [0.8, 0.2] # default is to calculate the average value of objectives
```

## 4.3 Config custom objective in code

Users can also register their own objective and pass it to quantizer as below:

```python
from neural_compressor.objective import Objective
from neural_compressor.experimental import Quantization

class CustomObj(Objective):
    representation = 'CustomObj'
    def __init__(self):
        super().__init__()
        # init code here

    def start(self):
        # do needed operators before inference

    def end(self):
        # do needed operators after the end of inference
        # add status value to self._result_list
        self._result_list.append(val)

quantizer = Quantization(yaml_file)
quantizer.objective = CustomObj()
quantizer.model = model
q_model = quantizer.fit()
```

