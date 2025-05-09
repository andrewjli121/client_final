# Metric Logger

Metric Logger (```metrics.MetricLogger```) is a tree-like data structure for organizing (arbitrary) recorded metrics during a benchmark.

## Overview
```MetricLogger``` is itself an abstract class for a tree node. ```MetricLogger``` is subclassed by specific types of nodes designed to record a particular type of metric. So far, we support:
- ```Container``` that stores nothing, but provides a section for children of other types.
- ```Timer``` to store time taken
- ```Value``` to store a single value such as an integer or string
- ```TokensPerSecondMeter``` to compute and store tokens per second generated by a language model.

## Using a MetricLogger

For the most part, you should interact with a ```MetricLogger``` through one of its subclasses. To define your own subclass, see [Extending to custom metrics](#extending-to-custom-metrics).

### Instantiation

Instantiation of a metric logger is done by calling the constructor for one of its subclasses. For example:

```python
from networked_robotics_benchmarking.metrics import Timer
timer = Timer("timer-name")
```

### Starting and Stopping Collection
All loggers and subclasses collect metrics between calls to ```start_collection``` and ```end_collection```. ```start_collection``` is automatically called by the constrcutor for a logger, but can be called again with no issues.

Example for a Timer:
```python
timer = Timer("name")
timer.start_collection()
time.sleep(1)
timer.end_collection()
```

For some subclasses, ```start_collection()``` and ```end_collection()``` can accept parameters.

Example for a Tokens Per Second Meter:
```python
tps = TokensPerSecondMeter("name")
tps.start_collection()
time.sleep(1)
tps.end_collection(num_tokens)
```

### Adding subsections

You can add subsections to your logger using the ```log_section()``` method.

```python
timer.log_section("initialization")
```

By default, ```log_section``` will create a node of the same type as its parent. To change this behavior, pass the node type as an argument to log_section:
```python
parent = Container("container-node")
child = parent.log_section("timer-node", Timer)
```

```log_section()``` returns a child logger, so the following is perfectly valid for creating (arbitrarily many) nested sections:

```python
timer = Timer("main-timer")
timer_a = timer.log_section("subsection")
timer_a.log_section("subsubsection")
```

### Context Managers
For some subclasses (such as ```Timer```), a context manager is implemented to simplify the syntax for timing code-blocks. Essentially, the context manager calls ```start_collection()``` and ```end_collection()``` itself.

```python
main_timer = Timer("main-timer-name")
with Timing(main_timer, "sub-timer-name"):
    do_stuff()
```

While context managers simplify syntax, they are not implemented for all node types, and for some node types, require users to end collection themselves. The following table shows which node types have context managers implemented.

| Node Type | Subclass Name | Context Manager Name | Context Manager Usage |
| --------- | ------------- | -------------------- | --------------------- |
| Container | ```Container``` | ```ContainerNode``` | Normal |
| Single Value | ```Value``` | ```ValueNode``` | User must call ```end_collection(value)``` |
| Timer | ```Timer``` | ```Timing``` | Normal |
| Tokens Per Second | ```TokensPerSecondMeter``` | ```TPSTracking``` | User must call ```end_collection(num_tokens)``` |

### Representations and Exporting Data
By default, the ```___str___()``` method for a metric logger is implemented to return a human-readable version.

To get a multi-indexed csv of the data, use the ```to_csv()``` method:
```python
timer = Timer("timer-name")
# Do some cool stuff, and time it all
timer.to_csv("csv_path")
```

Note: Don't use underscores in the timer-name. It won't convert to a csv correctly, since we use underscores as a delimiting character for multi-indexing.

To read the data from the csv back into a pandas dataframe, use the following snippet:
```python
pd.read_csv("csv_path", header = [0,1], index_col = 0)
```

## Extending to custom metrics
Extending a metric logger to track your own metric is simple, and involves just implementing two methods: ```_tic()``` and ```_toc()```, which are called by 
```start_collection()``` and ```end_collection()``` respectively.

Example:
```python
class MyCoolMetric(MetricLogger):
    def __init__(self, name, depth=0):
        super().__init__(name, depth)
        self.metric_name = "CoolMetric"
    
    def _tic(self):
        #Code that happens when starting collection

    def _toc(self):
        #Code to end collection
        return metric_value
```

If you need to add parameters to ```_tic()``` and ```_toc()```, it's as simple as including them in your subclass method definitions. Just make sure to call the user-facing functions ```start_collection()``` and ```end_collection()``` with the right arguments.